#!/usr/bin/env python
"""
TFPnP training, using the LIONsolver framework.

Usage:
    python -u scripts/train_tfpnp.py --output_dir results/learned/run_XX --n_epochs 50
"""

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader, TensorDataset

from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from ct_tfpnp.models.tfpnp_model import TFPnPModel
from ct_tfpnp.optimisers.tfpnp_solver import TFPnPSolver, TFPnPSolverParams


def build_model(args, geometry):
    """
    Build the TFPnP model with the CLI sigma/mu ranges.

    Args:
        args:     parsed CLI arguments.
        geometry: LION Geometry.

    Returns:
        TFPnPModel.
    """
    model_params = TFPnPModel.default_parameters()
    model_params.m = args.m
    model_params.sigma_min = args.sigma_floor
    model_params.sigma_max = args.sigma_ceil
    model_params.mu_min = args.mu_floor
    model_params.mu_max = args.mu_ceil
    return TFPnPModel(model_parameters=model_params, geometry=geometry)


def build_optimiser(model, args):
    """
    Adam over the Policy parameters only, with a lower learning rate on the pi_2 head.

    The critic is trained by a separate optimiser created inside TFPnPSolver.

    Args:
        model: TFPnPModel.
        args:  parsed CLI arguments.

    Returns:
        torch.optim.Adam with three parameter groups.
    """
    return torch.optim.Adam([
        {'params': model.policy.feature_extractor.parameters(), 'lr': args.lr_policy},
        {'params': model.policy.termination_head.parameters(), 'lr': args.lr_policy},
        {'params': model.policy.param_head.parameters(), 'lr': args.lr_pi2},
    ], lr=args.lr_policy)


def build_dataloader(train_images, batch_size=1):
    """
    Wrap the ground-truth images in a DataLoader.

    Args:
        train_images: list of ground-truth tensors, (1, H, W).
        batch_size:   images per loader batch.

    Returns:
        DataLoader yielding (dummy_sinogram, ground_truth).
    """
    # create dummy sinogram
    dummy_sino = torch.zeros(len(train_images), 1, 1, 1)
    dataset = TensorDataset(dummy_sino, torch.stack(train_images))

    # create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Device: {device} ({torch.cuda.get_device_name()})")

    # when set to 0, mixed noise is used
    noise_std = args.noise_std if args.noise_std > 0 else None
    print(f"Noise mode: {'mixed {5%, 7.5%, 10%}' if noise_std is None else f'fixed {noise_std*100:.1f}%'}")

    # load geometry
    geometry = experiment.experiment_params.geometry

    # build model
    model = build_model(args, geometry)
    print(f"  σ range: [{args.sigma_floor}, {args.sigma_ceil}] (8-bit equivalent: [{args.sigma_floor*5.1:.1f}, {args.sigma_ceil*5.1:.1f}])")
    print(f"  µ range: [{args.mu_floor}, {args.mu_ceil}]  (nominal; see admm.py)")

    # build optimiser
    optimizer = build_optimiser(model, args)

    # ── Solver params ─────────────────────────────────────────────────
    solver_params = TFPnPSolverParams(
        m=args.m, 
        N=args.N, 
        eta=args.eta,
        batch_size=args.batch_size,
        n_grad_steps=args.n_grad_steps,
        buffer_size=args.buffer_size,
        noise_std=noise_std,
        lr_critic=args.lr_critic,
        pi2_loss_scale=args.pi2_loss_scale,
        pi2_warmup_epochs=args.pi2_warmup,
        reward_type=args.reward_type,
        reward_alpha=args.reward_alpha,
        val_noise_std=args.val_noise_std,
        val_seed=args.val_seed,
        n_val_images=args.n_val,
    )

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

    # load data
    print("\nLoading LION dataset splits...")
    train_images = get_lion_split("train", geometry, device="cpu")
    val_images = get_lion_split("validation", geometry, device="cpu")

    if args.n_train > 0:
        train_images = train_images[:args.n_train]
    if args.n_val > 0:
        val_images = val_images[:args.n_val]

    print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

    # set up solver
    solver.set_training(build_dataloader(train_images))
    solver.set_validation(val_images)

    # set up saving and checkpointing
    solver.set_saving(save_folder, "tfpnp_final.pt")
    solver.set_checkpointing("tfpnp_check_*.pt", checkpoint_freq=5)

    # print training info
    print(f"\nStarting training: {args.n_epochs} epochs")
    print(f"  Train images:      {len(train_images)}")
    print(f"  Val images:        {len(val_images)} (fixed noise {args.val_noise_std*100:.1f}%, seed {args.val_seed})")
    print(f"  Buffer size:       {args.buffer_size}")
    print(f"  Batch (critic/π₁): {args.batch_size}")
    print(f"  Batch (π₂):        {args.pi2_batch_size}")
    print(f"  n_grad_steps:      {args.n_grad_steps}")
    print(f"  π₂ warmup:         {args.pi2_warmup} epochs")
    print(f"  π₂ loss scale:     {args.pi2_loss_scale}")
    print(f"  π₂ reward term:    {'on (eq. 17)' if not args.pi2_no_reward else 'OFF (V-only)'}")
    print(f"  Reward:            {args.reward_type}" + (f" (α={args.reward_alpha})" if args.reward_type != 'psnr' else ""))
    print(f"  LR policy: {args.lr_policy}, LR π₂: {args.lr_pi2}, LR critic: {args.lr_critic}")
    
    solver.train(args.n_epochs)

    # save final results
    solver.save_final_results()
    print(f"\nTraining complete. Saved to {save_folder}")
    if solver.best_val_psnr > -float('inf'):
        print(f"Best val PSNR: {solver.best_val_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFPnP with LIONsolver")
    parser.add_argument("--denoiser_path", type=str, default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    parser.add_argument("--output_dir", type=str, default="results/learned")
    parser.add_argument("--n_train", type=int, default=200, help="Max training images. 0 = use all (3300 from LION train split).")
    parser.add_argument("--n_val", type=int, default=30, help="Max validation images. 0 = use all (401 from LION val split).")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0, help="Noise fraction. 0 = random from {5%%, 7.5%%, 10%%}")
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8, help="Replay batch for the critic and π₁. MUST be > 1.")
    parser.add_argument("--pi2_batch_size", type=int, default=2, help="Replay batch for π₂. MUST be > 1.")
    parser.add_argument("--lr_policy", type=float, default=3e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--lr_pi2", type=float, default=1e-6)
    parser.add_argument("--pi2_warmup", type=int, default=5)
    parser.add_argument("--pi2_loss_scale", type=float, default=0.01)
    parser.add_argument("--pi2_no_reward", action="store_true", help="Drop the r(s,a) term from eq. 17.")
    parser.add_argument("--sigma_floor", type=float, default=1.0, help="Lower bound for policy's σ output")
    parser.add_argument("--sigma_ceil", type=float, default=5.0, help="Upper bound for policy's σ output")
    parser.add_argument("--mu_floor", type=float, default=10.0, help="Lower bound for policy's µ output")
    parser.add_argument("--mu_ceil", type=float, default=100.0, help="Upper bound for policy's µ output")
    
    parser.add_argument("--reward_type", type=str, default="psnr",
                        choices=["psnr", "psnr_ssim", "psnr_haarpsi", "ssim", "haarpsi"],
                        help="Reward function: ΔPSNR alone, +α·ΔSSIM, or +α·ΔHaarPSI")
    parser.add_argument("--reward_alpha", type=float, default=5.0,
                        help="Weight on SSIM/HaarPSI in reward (ignored if reward_type=psnr)")
    
    parser.add_argument("--val_noise_std", type=float, default=0.05,
                        help="Fixed noise level used for validation.")
    parser.add_argument("--val_seed", type=int, default=12345,
                        help="Fixed noise realisation, so val PSNR is comparable across epochs.")
    
    args = parser.parse_args()
    main(args)