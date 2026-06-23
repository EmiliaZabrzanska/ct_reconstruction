#!/usr/bin/env python3
"""
Train CT-specific DRUNet on LIDC images.

Drop-in replacement for natural-image DRUNet (drunet_gray.pth).
Same architecture (DRUNet class), same input convention (image + σ/50 map),
same checkpoint format (raw state_dict). Just swap denoiser_path to use it.

Usage:
    python -u scripts/train_drunet_ct.py \
        --output_dir results/baselines/drunet_ct \
        --n_epochs 50 \
        --batch_size 8
"""

import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ct_tfpnp.models.denoiser import DRUNet
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.experiments.parallel_beam_ct import experiment


def add_noise(clean, sigma_max=50.0):
    """Per-image [0,1] normalize, add Gaussian noise at random σ ∈ [0, σ_max].

    Returns (noisy_01, clean_01, sigma) where sigma is in DRUNet convention
    (where σ=50 corresponds to noise std = 50/255 in the [0,1] domain).
    """
    B = clean.shape[0]
    scale = clean.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
    clean_01 = clean / scale

    sigma = torch.rand(B, 1, 1, 1, device=clean.device) * sigma_max
    noise = torch.randn_like(clean_01) * (sigma / 255.0)
    noisy_01 = clean_01 + noise

    return noisy_01, clean_01, sigma


def make_drunet_input(noisy_01, sigma):
    """Concatenate noisy image with σ/50 map for DRUNet input — matches wrapper."""
    B, _, H, W = noisy_01.shape
    sigma_map = (sigma / 50.0).expand(B, 1, H, W)
    return torch.cat([noisy_01, sigma_map], dim=1)


def evaluate(model, val_loader, device, sigma_eval=15.0):
    """Validate: denoise at a fixed σ, report mean PSNR in [0,1] domain."""
    model.eval()
    psnrs = []
    with torch.no_grad():
        for batch in val_loader:
            clean = batch[0].to(device)
            B = clean.shape[0]
            scale = clean.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
            clean_01 = clean / scale

            sigma_fixed = torch.full((B, 1, 1, 1), sigma_eval, device=device)
            noise = torch.randn_like(clean_01) * (sigma_fixed / 255.0)
            noisy_01 = clean_01 + noise

            inp = make_drunet_input(noisy_01, sigma_fixed)
            pred = model(inp).clamp(0, 1)

            mse = ((pred - clean_01) ** 2).mean(dim=[1, 2, 3])
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
            psnrs.extend(psnr.cpu().tolist())

    return float(np.mean(psnrs))


def main(args):
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Parse multi-σ validation list
    val_sigmas = [float(s) for s in args.val_sigmas.split(",")]
    print(f"Validation σ values: {val_sigmas} "
          f"(≈ {', '.join(f'{s/255*100:.1f}%' for s in val_sigmas)} image-domain noise)")

    if args.best_sigma not in val_sigmas:
        raise ValueError(
            f"--best_sigma={args.best_sigma} must be in --val_sigmas={val_sigmas}"
        )
    print(f"Best-checkpoint criterion: PSNR at σ={args.best_sigma:g}")

    # ── Geometry — use the project's pre-built one (same as NB10/NB12) ──
    geo = experiment.experiment_params.geometry
    print(f"\nGeometry: {len(geo.angles)} angles, parallel beam")

    # ── Data ────────────────────────────────────────────────────────
    print("\nLoading LIDC splits...")
    train_images, _ = get_lion_split(split="train", geometry=geo, device="cpu")
    val_images, _ = get_lion_split(split="validation", geometry=geo, device="cpu")

    if args.n_train > 0:
        train_images = train_images[:args.n_train]
    if args.n_val > 0:
        val_images = val_images[:args.n_val]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_stack = torch.stack(train_images)
    val_stack = torch.stack(val_images)
    train_loader = DataLoader(TensorDataset(train_stack),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(TensorDataset(val_stack),
                            batch_size=args.batch_size, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────
    model = DRUNet(in_channels=2, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDRUNet: {n_params/1e6:.1f}M parameters")
    print(f"Training σ range: [0, {args.sigma_max}]")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # ── Training loop ───────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'val_psnr': {sv: [] for sv in val_sigmas},
        'best_val_psnr': -float('inf'),
        'best_epoch': -1,
        'best_sigma': args.best_sigma,
        'config': vars(args),
    }

    print(f"\nStarting training: {args.n_epochs} epochs")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            clean = batch[0].to(device)
            noisy_01, clean_01, sigma = add_noise(clean, sigma_max=args.sigma_max)
            inp = make_drunet_input(noisy_01, sigma)

            pred = model(inp)
            loss = ((pred - clean_01) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(mean_loss)

        # Validate at each σ level
        val_strs = []
        for sv in val_sigmas:
            val_psnr = evaluate(model, val_loader, device, sigma_eval=sv)
            history['val_psnr'][sv].append(val_psnr)
            val_strs.append(f"σ={sv:g}: {val_psnr:.2f}")

        # Best checkpoint based on the primary σ
        primary_psnr = history['val_psnr'][args.best_sigma][-1]
        new_best = ""
        if primary_psnr > history['best_val_psnr']:
            history['best_val_psnr'] = primary_psnr
            history['best_epoch'] = epoch
            # Save as raw state_dict — matches the wrapper's load_pretrained()
            torch.save(model.state_dict(), out_dir / "drunet_ct_best.pth")
            new_best = " ★"

        print(f"Epoch {epoch:3d}/{args.n_epochs}: "
              f"loss={mean_loss:.6f} | val PSNR {' | '.join(val_strs)} dB{new_best}")

        with open(out_dir / "drunet_ct_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    torch.save(model.state_dict(), out_dir / "drunet_ct_final.pth")
    print(f"\nDone.")
    print(f"  Best val PSNR (σ={args.best_sigma:g}): {history['best_val_psnr']:.2f} dB "
          f"(epoch {history['best_epoch']})")
    print(f"  Drop-in checkpoint: {out_dir / 'drunet_ct_best.pth'}")
    print(f"  Use with: --denoiser_path {out_dir.resolve() / 'drunet_ct_best.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default="results/baselines/drunet_ct")
    p.add_argument("--n_train", type=int, default=0,
                   help="Max training images. 0 = all (3300 from LION train split).")
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sigma_max", type=float, default=50.0,
                   help="Max σ during training (DRUNet convention).")
    p.add_argument("--val_sigmas", type=str, default="5,15,25",
                   help="Comma-separated list of σ values to validate at.")
    p.add_argument("--best_sigma", type=float, default=15.0,
                   help="σ used for best-checkpoint selection. Must be in --val_sigmas.")
    args = p.parse_args()
    main(args)