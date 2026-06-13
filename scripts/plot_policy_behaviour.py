#!/usr/bin/env python3
"""
Visualise the TFPnP policy's σ and µ trajectories across multiple validation images.

Mirrors Figure 8 of Wei et al. (2022): for each selected image, shows
ground truth + σ-vs-iteration + µ-vs-iteration in a 3xN grid.

Usage:
    python scripts/plot_policy_behaviour.py --experiment_name run_03_full
    python scripts/plot_policy_behaviour.py --experiment_name run_03_full --n_images 6 --noise_std 0.075
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import ct_tfpnp
from ct_tfpnp.models.policy import ResNetActor_ADMM
from ct_tfpnp.models.denoiser import DRUNetDenoiser
from ct_tfpnp.ct_ops.admm import ADMMStep
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.evaluation.metrics import psnr_np as psnr, ls_scale
from ct_tfpnp.datasets.lidc import get_lion_split, is_lung_slice
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from ct_tfpnp.utils import to_4d, read_metrics_config
from LION.CTtools.ct_utils import make_operator

OUTPUT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")


def load_policy(ckpt_dir, device):
    """Same loading logic as the other eval scripts."""
    ckpt_path = ckpt_dir / "checkpoint_best_val.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded: {ckpt_path.name}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val PSNR: {ckpt.get('val_psnr', 0):.2f} dB")

    cfg = read_metrics_config(ckpt_dir)
    sigma_range = tuple(cfg.get('sigma_range') or (1.0, 5.0))
    mu_range = tuple(cfg.get('mu_range') or (10.0, 100.0))
    print(f"  σ range: {sigma_range}")
    print(f"  µ range: {mu_range}")

    if 'model_state_dict' in ckpt:
        from ct_tfpnp.models.tfpnp_model import TFPnPModel
        mp = TFPnPModel.default_parameters()
        mp.sigma_min, mp.sigma_max = sigma_range
        mp.mu_min, mp.mu_max = mu_range
        model = TFPnPModel(model_parameters=mp,
                           geometry=experiment.experiment_params.geometry)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.policy.to(device).eval(), sigma_range, mu_range
    elif 'policy_state_dict' in ckpt:
        policy = ResNetActor_ADMM(in_channels=5, n_action_steps=5,
                                  sigma_range=sigma_range,
                                  mu_range=mu_range).to(device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        return policy.eval(), sigma_range, mu_range
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")


def run_tfpnp_recording(policy, admm_step, gt, op, noise_std, device, m, N, seed=99):
    """Run TFPnP on one image, recording every σ and µ chosen."""
    sino_clean = op.forward(gt)
    SCALE = sino_clean.max() / gt.max()
    torch.manual_seed(seed)
    sino_scaled = sino_clean / SCALE
    noise = noise_std * sino_scaled.std() * torch.randn_like(sino_scaled)
    y = (sino_scaled + noise) * SCALE

    x_fbp = lion_fbp(y, op)
    x_fbp = ls_scale(gt, x_fbp).clamp(min=0)
    fbp_psnr = psnr(gt, x_fbp)

    x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    noise_level = torch.tensor([noise_std], device=device)

    all_sigma, all_mu = [], []
    n_decisions = 0

    with torch.no_grad():
        for t in range(N):
            iter_frac = torch.tensor([t / N], device=device)
            stop_logits, sigma_seq, mu_seq = policy(
                to_4d(x), to_4d(z), to_4d(u), noise_level, iter_frac)
            if t > 0 and F.softmax(stop_logits, -1)[0, 1].item() > 0.5:
                break

            all_sigma.extend(sigma_seq[0].tolist())
            all_mu.extend(mu_seq[0].tolist())
            n_decisions += 1

            for i in range(m):
                x, z, u = admm_step(x, z, u, y,
                                    sigma=sigma_seq[0, i],
                                    mu=mu_seq[0, i])

    x_final = ls_scale(gt, x).clamp(min=0)
    return {
        'gt': gt,
        'x_fbp': x_fbp,
        'x_final': x_final,
        'sigmas': all_sigma,
        'mus': all_mu,
        'n_decisions': n_decisions,
        'fbp_psnr': fbp_psnr,
        'final_psnr': psnr(gt, x_final),
    }


def select_lung_images(val_images, n_images):
    """Pick n_images lung slices, evenly spaced across all lung indices."""
    lung_indices = [i for i, img in enumerate(val_images) if is_lung_slice(img)]
    if not lung_indices:
        print("Warning: no lung slices found, falling back to evenly-spaced full set")
        chosen = np.linspace(0, len(val_images) - 1, n_images, dtype=int)
        return list(chosen)
    chosen_positions = np.linspace(0, len(lung_indices) - 1, n_images, dtype=int)
    return [lung_indices[p] for p in chosen_positions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--n_images", type=int, default=5,
                   help="Number of validation images to show (default 5)")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--m", type=int, default=None,
               help="m (ADMM iters per decision). If None, read from JSON.")
    p.add_argument("--N", type=int, default=None,
                help="N (max decisions). If None, read from JSON.")
    p.add_argument("--denoiser_path", type=str,
                   default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")
    print(f"n_images   : {args.n_images}")
    print(f"noise      : {args.noise_std*100:.1f}%")

    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────
    geo = experiment.experiment_params.geometry
    op = make_operator(geo)
    denoiser = DRUNetDenoiser(pretrained_path=args.denoiser_path).to(device)
    for p_ in denoiser.parameters():
        p_.requires_grad_(False)
    denoiser.eval()
    admm_step = ADMMStep(op=op, denoiser=denoiser, n_x_steps=6)

    # ── Load policy ────────────────────────────────────────────────────
    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy, sigma_range, mu_range = load_policy(ckpt_dir, device)
    cfg = read_metrics_config(ckpt_dir)
    auto_m = int(cfg.get('m', 5))
    auto_N = int(cfg.get('N', 6))
    if args.m is None:
        args.m = auto_m
    if args.N is None:
        args.N = auto_N
    print(f"m, N       : {args.m}, {args.N}")

    # ── Pick images ────────────────────────────────────────────────────
    val_images, _ = get_lion_split("validation", geometry=geo, device=device)
    selected_indices = select_lung_images(val_images, args.n_images)
    print(f"Selected val indices: {selected_indices}")

    # ── Run TFPnP on each ──────────────────────────────────────────────
    all_data = []
    for idx in selected_indices:
        print(f"  Running on val_images[{idx}]...")
        data = run_tfpnp_recording(
            policy, admm_step, val_images[idx], op,
            args.noise_std, device, args.m, args.N)
        data['idx'] = idx
        all_data.append(data)
        print(f"    {data['n_decisions']} decisions, "
              f"FBP={data['fbp_psnr']:.1f} → TFPnP={data['final_psnr']:.1f} dB")

    # ── Plot ───────────────────────────────────────────────────────────
    n = len(all_data)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9),
                             gridspec_kw={'height_ratios': [1.4, 1, 1]})
    if n == 1:
        axes = axes.reshape(3, 1)

    sigma_color = '#1F4E79'
    mu_color = '#C00000'

    for col, data in enumerate(all_data):
        # Row 1: ground truth
        ax = axes[0, col]
        gt_np = data['gt'][0].cpu().numpy()
        ax.imshow(gt_np, cmap='gray', vmin=gt_np.min(), vmax=gt_np.max())
        ax.set_title(f"Val #{data['idx']}\n"
                     f"FBP={data['fbp_psnr']:.1f} → TFPnP={data['final_psnr']:.1f} dB",
                     fontsize=9)
        ax.axis('off')

        # Row 2: σ trajectory
        ax = axes[1, col]
        iterations = np.arange(1, len(data['sigmas']) + 1)
        ax.plot(iterations, data['sigmas'], 'o-',
                color=sigma_color, lw=1.4, ms=5, alpha=0.85)
        for d in range(1, data['n_decisions']):
            ax.axvline(d * args.m + 0.5, color='gray', ls=':', alpha=0.4, lw=0.8)
        ax.set_xlabel('ADMM iteration', fontsize=8)
        if col == 0:
            ax.set_ylabel('σ', fontsize=10, color=sigma_color)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(sigma_range[0] * 0.8, sigma_range[1] * 1.05)
        ax.axhline(sigma_range[0], color=sigma_color, ls='--', alpha=0.25, lw=0.8)
        ax.axhline(sigma_range[1], color=sigma_color, ls='--', alpha=0.25, lw=0.8)

        # Row 3: µ trajectory
        ax = axes[2, col]
        ax.plot(iterations, data['mus'], 'o-',
                color=mu_color, lw=1.4, ms=5, alpha=0.85)
        for d in range(1, data['n_decisions']):
            ax.axvline(d * args.m + 0.5, color='gray', ls=':', alpha=0.4, lw=0.8)
        ax.set_xlabel('ADMM iteration', fontsize=8)
        if col == 0:
            ax.set_ylabel('µ', fontsize=10, color=mu_color)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(mu_range[0] * 0.8, mu_range[1] * 1.05)
        ax.axhline(mu_range[0], color=mu_color, ls='--', alpha=0.25, lw=0.8)
        ax.axhline(mu_range[1], color=mu_color, ls='--', alpha=0.25, lw=0.8)

    plt.suptitle(
        f"Learned Policy Behaviour — {args.experiment_name} "
        f"(σ ∈ {sigma_range}, µ ∈ {mu_range}) at {args.noise_std*100:.1f}% noise",
        fontsize=11, y=1.0)
    plt.tight_layout()

    out_path = output_dir / "policy_behaviour.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()