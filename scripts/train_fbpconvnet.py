#!/usr/bin/env python
"""Train FBPConvNet (parallel-beam variant) on LIDC sparse-view CT."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import ct_tfpnp
from ct_tfpnp.models.fbpconvnet_image import FBPConvNetImage
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.evaluation.metrics import psnr_np as psnr, ls_scale
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.utils import setup_admm, to_4d
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from LION.CTtools.ct_utils import make_operator
from LION.models.post_processing.FBPConvNet import FBPConvNet


def main(args):
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    geo = experiment.experiment_params.geometry
    op = make_operator(geo)

    # Data
    print("Loading LION splits...")
    train_images, _ = get_lion_split("train", geo, device="cpu")
    val_images, _ = get_lion_split("validation", geo, device="cpu")
    if args.n_train > 0:
        train_images = train_images[:args.n_train]
    if args.n_val > 0:
        val_images = val_images[:args.n_val]
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_ds = TensorDataset(
        torch.zeros(len(train_images), 1, 1, 1),
        torch.stack(train_images))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model_params = FBPConvNet.default_parameters()
    model = FBPConvNetImage(geometry_parameters=geo,
                            model_parameters=model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_psnrs = [], []
    best_val = -float('inf')

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _, gt_batch in train_loader:
            gt_batch = gt_batch.to(device)

            with torch.no_grad():
                fbp_b_list = []
                for g in gt_batch:
                    sino_b = op.forward(g)
                    scale_b = sino_b.max() / g.max()
                    noise_b = args.noise_std * (sino_b / scale_b).std() * torch.randn_like(sino_b)
                    y_b = (sino_b / scale_b + noise_b) * scale_b
                    fbp_b_list.append(lion_fbp(y_b, op))
                fbp_b = torch.stack(fbp_b_list)

            optimizer.zero_grad()
            out = model(fbp_b)
            loss = loss_fn(out, gt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_psnr_sum = 0.0
        with torch.no_grad():
            for v_gt in val_images:
                v_gt = v_gt.to(device)
                sino_v = op.forward(v_gt)
                scale_v = sino_v.max() / v_gt.max()
                torch.manual_seed(42)
                noise_v = args.noise_std * (sino_v / scale_v).std() * torch.randn_like(sino_v)
                y_v = (sino_v / scale_v + noise_v) * scale_v
                fbp_v = lion_fbp(y_v, op)
                out_v = model(to_4d(fbp_v))
                val_psnr_sum += psnr(v_gt, ls_scale(v_gt, out_v[0]).clamp(min=0))
        avg_val = val_psnr_sum / len(val_images)
        val_psnrs.append(avg_val)

        if avg_val > best_val:
            best_val = avg_val
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_psnr': avg_val},
                       save_dir / "checkpoint_best_val.pth")

        print(f"Epoch {epoch:>3}/{args.n_epochs}: loss={avg_loss:.6f} | "
              f"val PSNR={avg_val:.2f} dB"
              + ("  (new best)" if avg_val == best_val else ""))

    # Save training history for plot_training_curves.py compatibility
    import json
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
    parser.add_argument("--noise_std", type=float, default=0.05)
    args = parser.parse_args()
    main(args)