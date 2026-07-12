#!/usr/bin/env python3
"""
Visualise the TFPnP policy's sigma and mu trajectories across validation images.

Usage:
    python scripts/plot_policy_behaviour.py --experiment_name run_03_full
    python scripts/plot_policy_behaviour.py --experiment_name run_03_full --n_images 6 --noise_std 0.075
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

import ct_tfpnp
from ct_tfpnp.datasets.lidc import get_lion_split, is_lung_slice
from ct_tfpnp.evaluation.metrics import ls_scale, psnr_np as psnr
from ct_tfpnp.training.environment import run_episode
from ct_tfpnp.utils import read_metrics_config, setup_admm

from evaluate_run import CHECKPOINT_BASE, M_DEFAULT, N_DEFAULT, OUTPUT_BASE, load_checkpoint

# fixed noise
BEHAVIOUR_SEED = 99

SIGMA_COLOR = '#1F4E79'
MU_COLOR = '#C00000'


def select_lung_images(val_images, n_images):
    """
    Pick n_images lung slices, evenly spaced across all lung indices.

    Args:
        val_images: list of ground-truth tensors.
        n_images:   how many to select.

    Returns:
        List of indices into val_images.
    """
    lung_indices = [i for i, img in enumerate(val_images) if is_lung_slice(img)]
    
    if not lung_indices:
        print("Warning: no lung slices found, falling back to evenly-spaced full set")
        chosen = np.linspace(0, len(val_images) - 1, n_images, dtype=int)
        return list(chosen)
    
    chosen_positions = np.linspace(0, len(lung_indices) - 1, n_images, dtype=int)
    
    return [lung_indices[p] for p in chosen_positions]


def record_policy(policy, admm_step, gt, op, noise_std, m, N):
    """
    Run TFPnP on one image and return everything needed to plot its behaviour.

    Args:
        policy:    trained policy, in eval() mode.
        admm_step: ADMMStep.
        gt:        ground truth, (1, H, W).
        op:        LION CT operator.
        noise_std: fractional noise level.
        m, N:      rollout budget.

    Returns:
        dict with 'gt', 'x_fbp', 'x_final', 'sigmas', 'mus', 'n_decisions',
        'fbp_psnr' and 'final_psnr'. The PSNRs use ls_scale for reporting only.
    """
    # run episode with fixed seed
    ep = run_episode(policy, 
                     admm_step, 
                     gt, 
                     op,
                     noise_std=noise_std, 
                     m=m, 
                     N=N,
                     stochastic=False, 
                     seed=BEHAVIOUR_SEED
    )
    # scale outputs for reporting 
    x_fbp = ls_scale(gt, ep['x0']).clamp(min=0)
    x_final = ls_scale(gt, ep['x']).clamp(min=0)

    # return data for plotting
    return {
        'gt': gt,
        'x_fbp': x_fbp,
        'x_final': x_final,
        'sigmas': ep['sigmas'],
        'mus': ep['mus'],
        'n_decisions': ep['n_decisions'],
        'fbp_psnr': psnr(gt, x_fbp),
        'final_psnr': psnr(gt, x_final),
    }


def plot_trajectory(ax, values, n_decisions, m, color, ylabel, value_range, show_ylabel):
    """
    Plot one parameter trajectory against ADMM iteration.

    Args:
        ax:           matplotlib axis.
        values:       the parameter value at every ADMM iteration.
        n_decisions:  number of policy decisions taken.
        m:            ADMM steps per decision.
        color:        line colour.
        ylabel:       axis label.
        value_range:  (min, max) admissible range.
        show_ylabel:  whether to draw the y label (only on the leftmost column).
    """
    iterations = np.arange(1, len(values) + 1)
    ax.plot(iterations, values, 'o-', color=color, lw=1.4, ms=5, alpha=0.85)

    # set decision boundaries
    for d in range(1, n_decisions):
        ax.axvline(d * m + 0.5, color='gray', ls=':', alpha=0.4, lw=0.8)

    # set range lines
    ax.axhline(value_range[0], color=color, ls='--', alpha=0.25, lw=0.8)
    ax.axhline(value_range[1], color=color, ls='--', alpha=0.25, lw=0.8)
    ax.set_ylim(value_range[0] * 0.8, value_range[1] * 1.05)

    ax.set_xlabel('ADMM iteration', fontsize=8)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=color)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--n_images", type=int, default=5, help="Number of validation images to show (default 5)")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--m", type=int, default=None, help="m (ADMM iters per decision). If None, read from JSON.")
    p.add_argument("--N", type=int, default=None, help="N (max decisions). If None, read from JSON.")
    p.add_argument("--denoiser_path", type=str, default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")
    print(f"n_images   : {args.n_images}")
    print(f"noise      : {args.noise_std*100:.1f}%")

    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # set geometry, operator, denoiser, and ADMM step
    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)

    # load trained policy from checkpoint
    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy = load_checkpoint(ckpt_dir, device)

    # load m and N from metrics_history.json
    cfg = read_metrics_config(ckpt_dir)
    if args.m is None:
        args.m = int(cfg.get('m', M_DEFAULT))
    if args.N is None:
        args.N = int(cfg.get('N', N_DEFAULT))
    sigma_range = tuple(cfg.get('sigma_range') or (1.0, 5.0))
    mu_range = tuple(cfg.get('mu_range') or (10.0, 100.0))
    print(f"m, N       : {args.m}, {args.N}")

    # pick images
    val_images = get_lion_split("validation", geometry=geo, device=device)
    selected_indices = select_lung_images(val_images, args.n_images)
    print(f"Selected val indices: {selected_indices}")

    # run TFPnP on each image
    all_data = []
    for idx in selected_indices:
        print(f"  Running on val_images[{idx}]...")
        data = record_policy(policy, admm_step, val_images[idx], op, args.noise_std, args.m, args.N)
        data['idx'] = idx
        all_data.append(data)
        print(f"    {data['n_decisions']} decisions, FBP={data['fbp_psnr']:.1f} → TFPnP={data['final_psnr']:.1f} dB")

    # plot 3 rows with 1 col per image
    n = len(all_data)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9), gridspec_kw={'height_ratios': [1.4, 1, 1]})
    
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, data in enumerate(all_data):
        # ground truth
        ax = axes[0, col]
        gt_np = data['gt'][0].cpu().numpy()
        ax.imshow(gt_np, cmap='gray', vmin=gt_np.min(), vmax=gt_np.max())
        ax.set_title(f"Val #{data['idx']}\n FBP={data['fbp_psnr']:.1f} → TFPnP={data['final_psnr']:.1f} dB", fontsize=9)
        ax.axis('off')

        # sigma trajectory
        plot_trajectory(axes[1, col], data['sigmas'], data['n_decisions'], args.m, SIGMA_COLOR, 'σ', sigma_range, show_ylabel=(col == 0))

        # mu trajectory 
        plot_trajectory(axes[2, col], data['mus'], data['n_decisions'], args.m, MU_COLOR, 'µ (nominal)', mu_range, show_ylabel=(col == 0))

    plt.suptitle(f"Learned Policy Behaviour — {args.experiment_name} (σ ∈ {sigma_range}, µ ∈ {mu_range}) at {args.noise_std*100:.1f}% noise", fontsize=11, y=1.0)
    plt.tight_layout()

    out_path = output_dir / "policy_behaviour.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()