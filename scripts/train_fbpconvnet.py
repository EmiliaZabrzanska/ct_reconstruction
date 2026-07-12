#!/usr/bin/env python
"""
Train FBPConvNet (parallel-beam variant) on LIDC sparse-view CT.

Usage:
    python -u scripts/train_fbpconvnet.py --output_dir results/learned/fbpconvnet_pat_250_e80
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import ct_tfpnp
from ct_tfpnp.ct_ops.fbp import calibrate_to_data, fbp as lion_fbp
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.evaluation.metrics import ls_scale, psnr_np as psnr
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from ct_tfpnp.models.fbpconvnet_image import FBPConvNetImage
from ct_tfpnp.utils import project_and_add_noise, to_4d
from LION.CTtools.ct_utils import make_operator
from LION.models.post_processing.FBPConvNet import FBPConvNet

# add mixed level noise 
NOISE_LEVELS = (0.05, 0.075, 0.10)

# set vsalidation seed
VAL_SEED = 42


def make_fbp(gt, op, noise_std, seed=None):
    """
    Project one ground-truth image, add noise, and reconstruct with FBP.

    Args:
        gt:        ground truth, (1, H, W).
        op:        LION CT operator.
        noise_std: fractional sinogram noise level.
        seed:      optional seed for the noise (local RNG only).

    Returns:
        The calibrated FBP reconstruction, (1, H, W).
    """
    # simulate noisy sinogram
    y = project_and_add_noise(gt, op, noise_std, seed=seed)

    # calibrate FBP to the noisy sinogram
    return calibrate_to_data(lion_fbp(y, op), y, op)


def build_fbp_batch(gt_batch, op, noise_std):
    """
    Build a batch of FBP inputs, one noise level per image.

    Args:
        gt_batch:  ground-truth batch, (B, 1, H, W).
        op:        LION CT operator.
        noise_std: fixed noise level, or 0 to sample per image from NOISE_LEVELS.

    Returns:
        FBP batch, (B, 1, H, W).
    """
    with torch.no_grad():
        fbps = []
        for gt in gt_batch:
            level = (float(np.random.choice(NOISE_LEVELS)) if noise_std == 0 else noise_std)
            fbps.append(make_fbp(gt, op, level))
        return torch.stack(fbps)


def validate(model, val_images, op, noise_levels, device):
    """
    Evaluate at each noise level separately, with fixed noise realisations.

    Args:
        model:        FBPConvNetImage.
        val_images:   list of ground-truth tensors.
        op:           LION CT operator.
        noise_levels: levels to evaluate at.
        device:       torch device.

    Returns:
        {noise_level: mean PSNR}
    """
    # set model to eval mode
    model.eval()
    by_level = {}

    # evaluate each noise level separately
    with torch.no_grad():
        for level in noise_levels:
            psnr_sum = 0.0
            for v_idx, v_gt in enumerate(val_images):
                v_gt = v_gt.to(device)
                fbp_v = make_fbp(v_gt, op, level, seed=VAL_SEED + v_idx * 100)
                out_v = model(to_4d(fbp_v))
                psnr_sum += psnr(v_gt, ls_scale(v_gt, out_v[0]).clamp(min=0))
            by_level[level] = psnr_sum / len(val_images)

    return by_level


def main(args):
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    # make geometry and operator
    geo = experiment.experiment_params.geometry
    op = make_operator(geo)

    # load data
    print("Loading LION splits...")
    train_images = get_lion_split("train", geo, device="cpu")
    val_images = get_lion_split("validation", geo, device="cpu")
    if args.n_train > 0:
        train_images = train_images[:args.n_train]
    if args.n_val > 0:
        val_images = val_images[:args.n_val]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # build training dataset and loader
    train_ds = TensorDataset(torch.zeros(len(train_images), 1, 1, 1), torch.stack(train_images))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # build model
    model_params = FBPConvNet.default_parameters()
    model = FBPConvNetImage(geometry_parameters=geo, model_parameters=model_params).to(device)
    
    # build optimiser and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # validation noise levels
    val_noise_levels = list(NOISE_LEVELS) if args.noise_std == 0 else [args.noise_std]
    train_losses, val_psnrs = [], []
    best_val = -float('inf')

    # training loop
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _, gt_batch in train_loader:
            gt_batch = gt_batch.to(device)

            # build FBP batch with noise
            fbp_b = build_fbp_batch(gt_batch, op, args.noise_std)

            optimizer.zero_grad()
            loss = loss_fn(model(fbp_b), gt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validate per noise level
        by_level = validate(model, val_images, op, val_noise_levels, device)
        avg_val = sum(by_level.values()) / len(by_level)
        val_psnrs.append(avg_val)

        if avg_val > best_val:
            best_val = avg_val
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_psnr': avg_val},
                       save_dir / "checkpoint_best_val.pth")
            marker = "  (new best)"
        else:
            marker = ""

        # Print each level too for diagnostic visibility
        level_str = " | ".join(f"{nl*100:.1f}%: {p:.2f}" for nl, p in by_level.items())

        print(f"Epoch {epoch:>3}/{args.n_epochs}: loss={avg_loss:.6f} | "
              f"val PSNR avg={avg_val:.2f} dB ({level_str}){marker}")

    # save history for plotting
    history = {
        'train_loss': train_losses,
        'val_psnr': val_psnrs,
        'config': {
            'n_train': len(train_images),
            'n_val': len(val_images),
            'n_epochs': args.n_epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'noise_std': args.noise_std,
            'best_val_psnr': best_val,
        }
    }
    with open(save_dir / "metrics_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val PSNR: {best_val:.2f} dB → {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_train", type=int, default=250)
    parser.add_argument("--n_val", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise_std", type=float, default=0.0, help="Noise level. 0 = mixed sampling from {5%, 7.5%, 10%}")
    args = parser.parse_args()
    main(args)