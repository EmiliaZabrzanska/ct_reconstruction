#!/usr/bin/env python
"""TFPnP training script using LIONsolver framework."""

import torch
import numpy as np
import math
import argparse
import pathlib

from torch.utils.data import DataLoader, TensorDataset
from LION.CTtools.ct_geometry import Geometry
from ct_tfpnp.models.tfpnp_model import TFPnPModel
from ct_tfpnp.optimisers.tfpnp_solver import TFPnPSolver, TFPnPSolverParams
from ct_tfpnp.datasets.lidc import get_lion_split


def main(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Device: {device} ({torch.cuda.get_device_name()})")

    # Handle noise_std: 0 means random sampling (None)
    noise_std = args.noise_std if args.noise_std > 0 else None
    print(f"Noise mode: {'mixed {5%, 7.5%, 10%}' if noise_std is None else f'fixed {noise_std*100:.1f}%'}")

    # ── Geometry ──────────────────────────────────────────────────────
    n_det = math.ceil(512 * math.sqrt(2))
    geometry = Geometry(
        image_shape=[1, 512, 512], image_size=[1, 512, 512],
        detector_shape=[1, n_det], detector_size=[1, n_det],
        dso=1024, dsd=2048, mode="parallel",
        angles=np.linspace(0, np.pi, 30, endpoint=False),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = TFPnPModel(geometry=geometry)

    # ── Optimizer — separate LR for π₂ param_head ────────────────────
    optimizer = torch.optim.Adam([
        {'params': model.policy.feature_extractor.parameters(), 'lr': args.lr_policy},
        {'params': model.policy.termination_head.parameters(), 'lr': args.lr_policy},
        {'params': model.policy.param_head.parameters(), 'lr': args.lr_pi2},
    ], lr=args.lr_policy)

    # ── Solver params ─────────────────────────────────────────────────
    solver_params = TFPnPSolverParams(
        m=args.m, N=args.N, eta=args.eta,
        batch_size=args.batch_size,
        n_grad_steps=args.n_grad_steps,
        buffer_size=args.buffer_size,
        noise_std=noise_std,
        lr_critic=args.lr_critic,
        pi2_loss_scale=args.pi2_loss_scale,
        pi2_warmup_epochs=args.pi2_warmup,
    )

    # ── Solver ────────────────────────────────────────────────────────
    save_folder = pathlib.Path(args.output_dir)
    save_folder.mkdir(parents=True, exist_ok=True)

    solver = TFPnPSolver(
        model=model,
        optimizer=optimizer,
        geometry=geometry,
        verbose=True,
        device=device,
        solver_params=solver_params,
        save_folder=save_folder,
        denoiser_path=args.denoiser_path,
    )

    # ── Data (LION splits) ────────────────────────────────────────────
    print("\nLoading LION dataset splits...")
    train_images, _ = get_lion_split("train", geometry, device="cpu")
    val_images, _ = get_lion_split("validation", geometry, device="cpu")

    # Subset for practical training time
    if args.n_train is not None:
        train_images = train_images[:args.n_train]
    val_images = val_images[:args.n_val]

    print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

    gt_stack = torch.stack(train_images)
    dummy_sino = torch.zeros(len(train_images), 1, 1, 1)
    dataset = TensorDataset(dummy_sino, gt_stack)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    solver.set_training(train_loader)
    solver.set_validation(val_images)

    # ── Saving / checkpointing ────────────────────────────────────────
    solver.set_saving(save_folder, "tfpnp_final.pt")
    solver.set_checkpointing("tfpnp_check_*.pt", checkpoint_freq=5)

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nStarting training: {args.n_epochs} epochs")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images:   {len(val_images)}")
    print(f"  π₂ warmup:    {args.pi2_warmup} epochs")
    print(f"  π₂ loss scale: {args.pi2_loss_scale}")
    print(f"  LR policy: {args.lr_policy}, LR π₂: {args.lr_pi2}, LR critic: {args.lr_critic}")
    solver.train(args.n_epochs)

    # ── Save final ────────────────────────────────────────────────────
    solver.save_final_results()
    print(f"\nTraining complete. Saved to {save_folder}")
    if solver.best_val_psnr > -float('inf'):
        print(f"Best val PSNR: {solver.best_val_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFPnP with LIONsolver")
    parser.add_argument("--denoiser_path", type=str,
                        default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    parser.add_argument("--output_dir", type=str, default="results/learned")
    parser.add_argument("--n_train", type=int, default=200,
                        help="Max training images from LION split (None=all 3300)")
    parser.add_argument("--n_val", type=int, default=30,
                        help="Max validation images")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0,
                        help="Noise fraction. 0 = random from {5%%, 7.5%%, 10%%}")
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--lr_policy", type=float, default=3e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--lr_pi2", type=float, default=1e-6)
    parser.add_argument("--pi2_warmup", type=int, default=5)
    parser.add_argument("--pi2_loss_scale", type=float, default=0.01)
    parser.add_argument("--n_grad_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)