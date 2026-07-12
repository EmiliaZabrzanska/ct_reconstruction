#!/usr/bin/env python3
"""
Train a CT-specific DRUNet on LIDC images, as a drop-in replacement for the
natural-image drunet_gray.pth.

Usage:
    python -u scripts/train_drunet_ct.py \
        --output_dir results/baselines/drunet_ct \
        --n_epochs 50 --batch_size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from ct_tfpnp.models.denoiser import DRUNet


def normalise_01(clean):
    """
    Scale each image in the batch to [0, 1] by its own maximum.

    Args:
        clean: ground-truth batch, (B, 1, H, W), non-negative.

    Returns:
        (clean_01, scale) where clean = clean_01 * scale.
    """
    # find shape of image
    B = clean.shape[0]

    # scale by max val
    scale = clean.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)

    # return normalised image and scale factor
    return clean / scale, scale


def add_noise(clean, sigma_max, noise_map_scale):
    """
    Normalise to [0, 1] and add Gaussian noise at a random sigma.

    Args:
        clean:           ground-truth batch, (B, 1, H, W).
        sigma_max:       maximum sigma to sample (KAIR trains over [0, 50]).
        noise_map_scale: divisor turning sigma into a noise standard deviation.

    Returns:
        (noisy_01, clean_01, sigma) with sigma of shape (B, 1, 1, 1).
    """
    # obtain normalised image
    clean_01, _ = normalise_01(clean)

    # find batch shape
    B = clean.shape[0]

    # find random sigma for each image in the batch
    sigma = torch.rand(B, 1, 1, 1, device=clean.device) * sigma_max

    # add noise to the normalised image
    noisy_01 = clean_01 + torch.randn_like(clean_01) * (sigma / noise_map_scale)

    return noisy_01, clean_01, sigma


def make_drunet_input(noisy_01, sigma, noise_map_scale):
    """
    Concatenate the noisy image with its constant noise-level map.

    Args:
        noisy_01:        noisy image in [0, 1], (B, 1, H, W).
        sigma:           noise level, (B, 1, 1, 1).
        noise_map_scale: divisor, matching `add_noise`.

    Returns:
        (B, 2, H, W) tensor: image channel plus noise-level map.
    """
    # set shapes of batch and image
    B, _, H, W = noisy_01.shape

    # generate noise level map
    sigma_map = (sigma / noise_map_scale).expand(B, 1, H, W)

    # concatenate noisy image and noise map along channel dimension
    return torch.cat([noisy_01, sigma_map], dim=1)


def evaluate(model, val_loader, device, sigma_eval, noise_map_scale):
    """
    Denoise the validation set at one fixed sigma and return the mean PSNR.

    Args:
        model:           DRUNet.
        val_loader:      DataLoader over ground-truth images.
        device:          torch device.
        sigma_eval:      sigma to denoise at.
        noise_map_scale: divisor, matching training.

    Returns:
        Mean PSNR in dB.
    """
    # set model to evaluation mode
    model.eval()
    psnrs = []

    with torch.no_grad():

        # normalise clean images
        for batch in val_loader:
            clean = batch[0].to(device)
            clean_01, _ = normalise_01(clean)

            # add noise at the fixed sigma
            B = clean.shape[0]
            sigma_fixed = torch.full((B, 1, 1, 1), sigma_eval, device=device)
            noisy_01 = clean_01 + torch.randn_like(clean_01) * (sigma_fixed / noise_map_scale)

            # concatenate noisy image with noise-level map and denoise
            pred = model(make_drunet_input(noisy_01, sigma_fixed, noise_map_scale))
            pred = pred.clamp(0, 1)

            # compute PSNR for each image in the batch
            mse = ((pred - clean_01) ** 2).mean(dim=[1, 2, 3])
            psnrs.extend((10 * torch.log10(1.0 / mse.clamp(min=1e-10))).cpu().tolist())

    return float(np.mean(psnrs))


def main(args):

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Parse multi-σ validation list
    val_sigmas = [float(s) for s in args.val_sigmas.split(",")]
    print(f"Validation σ values: {val_sigmas} (≈ {', '.join(f'{s/args.noise_map_scale*100:.1f}%' for s in val_sigmas)} image-domain noise)")

    if args.best_sigma not in val_sigmas:
        raise ValueError(f"--best_sigma={args.best_sigma} must be in --val_sigmas={val_sigmas}")
    
    print(f"Best-checkpoint criterion: PSNR at σ={args.best_sigma:g}")

    # set geometry from the experiment
    geo = experiment.experiment_params.geometry
    print(f"\nGeometry: {len(geo.angles)} angles, parallel beam")

    # load data
    print("\nLoading LIDC splits...")
    train_images = get_lion_split(split="train", geometry=geo, device="cpu")
    val_images = get_lion_split(split="validation", geometry=geo, device="cpu")

    if args.n_train > 0:
        train_images = train_images[:args.n_train]
    if args.n_val > 0:
        val_images = val_images[:args.n_val]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_loader = DataLoader(TensorDataset(torch.stack(train_images)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.stack(val_images)), batch_size=args.batch_size, shuffle=False)

    # load model
    model = DRUNet(in_channels=2, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nDRUNet: {n_params/1e6:.1f}M parameters")
    print(f"Training σ range: [0, {args.sigma_max}]")

    # load optimiser and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

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

    # training loop
    print(f"\nStarting training: {args.n_epochs} epochs")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:

            # prepare input and target
            clean = batch[0].to(device)
            noisy_01, clean_01, sigma = add_noise(clean, args.sigma_max, args.noise_map_scale)
            inp = make_drunet_input(noisy_01, sigma, args.noise_map_scale)

            # compute loss
            loss = ((model(inp) - clean_01) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        mean_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(mean_loss)

        # validate at sigma values
        val_strs = []
        for sv in val_sigmas:
            val_psnr = evaluate(model, val_loader, device, sv, args.noise_map_scale)
            history['val_psnr'][sv].append(val_psnr)
            val_strs.append(f"σ={sv:g}: {val_psnr:.2f}")

        # save best checkpoint if primary PSNR is improved
        primary_psnr = history['val_psnr'][args.best_sigma][-1]
        new_best = ""
        if primary_psnr > history['best_val_psnr']:
            history['best_val_psnr'] = primary_psnr
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), out_dir / "drunet_ct_best.pth")
            new_best = " ★"

        print(f"Epoch {epoch:3d}/{args.n_epochs}: loss={mean_loss:.6f} | val PSNR {' | '.join(val_strs)} dB{new_best}")

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
    p.add_argument("--output_dir", type=str, default="results/baselines/drunet_ct")
    p.add_argument("--n_train", type=int, default=0, help="Max training images. 0 = all (3300 from LION train split).")
    p.add_argument("--n_val", type=int, default=50)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sigma_max", type=float, default=50.0, help="Max σ during training (DRUNet convention).")
    p.add_argument("--noise_map_scale", type=float, default=255.0, help="Divisor for BOTH the noise std and the conditioning map to match KAIR convention.")
    p.add_argument("--val_sigmas", type=str, default="5,15,25", help="Comma-separated list of σ values to validate at.")
    p.add_argument("--best_sigma", type=float, default=15.0, help="σ used for best-checkpoint selection. Must be in --val_sigmas.")
    args = p.parse_args()
    main(args)