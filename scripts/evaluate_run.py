#!/usr/bin/env python3
"""
Full evaluation script — produces all NB10 figures for a given experiment.

Usage:
    python scripts/evaluate_run.py --experiment_name run_02_pat_200
    python scripts/evaluate_run.py --experiment_name run_02_pat_200 --n_test_subset 50
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ct_tfpnp
from ct_tfpnp.models.policy import ResNetActor_ADMM
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.ct_ops.tv import tv_reconstruction
from ct_tfpnp.evaluation.metrics import evaluate_reconstruction, psnr_np as psnr, ls_scale
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.utils import to_4d, read_metrics_config, setup_admm

# ── Output paths (edit these once if your layout ever changes) ─────────
OUTPUT_BASE  = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
RESULTS_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/metrics")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")

# ── Constants ──────────────────────────────────────────────────────────
METHODS = ['FBP', 'TV', 'Fixed PnP-ADMM', 'TFPnP']
METHOD_LABELS = ['FBP', 'TV', 'Fixed\nPnP-ADMM', 'TFPnP\n(Ours)']
COLORS = ['#bdc3c7', '#f39c12', '#3498db', '#e74c3c']
METRIC_NAMES = ['psnr', 'ssim', 'haarpsi']
METRIC_LABELS = ['PSNR (dB)', 'SSIM', 'HaarPSI']
FIXED_SIGMA = 1.5
FIXED_MU = 20.0

def compute_metrics(gt, recon):
    gt_4d = to_4d(gt)
    recon_4d = to_4d(recon).clamp(0, float(gt.max()))
    return evaluate_reconstruction(recon_4d, gt_4d, float(gt.max()))

def load_checkpoint(ckpt_dir, device):
    ckpt_val = ckpt_dir / "checkpoint_best_val.pth"
    ckpt_best = ckpt_dir / "checkpoint_best.pth"
    lion_ckpts = sorted(ckpt_dir.glob("tfpnp_check_*.pt"))

    if ckpt_val.exists():
        ckpt = torch.load(ckpt_val, map_location=device, weights_only=False)
        print(f"Loaded {ckpt_val.name}: epoch {ckpt.get('epoch', '?')}, "
              f"val={ckpt.get('val_psnr', 0):.2f} dB")
    elif ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
        print(f"Loaded {ckpt_best.name}: epoch {ckpt['epoch']}")
    elif lion_ckpts:
        ckpt = torch.load(lion_ckpts[-1], map_location=device, weights_only=False)
        print(f"Loaded {lion_ckpts[-1].name}: epoch {ckpt['epoch']}")
    else:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    cfg = read_metrics_config(ckpt_dir)
    sigma_range = tuple(cfg.get('sigma_range') or (1.0, 5.0))
    mu_range = tuple(cfg.get('mu_range') or (10.0, 100.0))
    print(f"  σ range: {sigma_range}")
    print(f"  µ range: {mu_range}")

    if 'model_state_dict' in ckpt:
        from ct_tfpnp.models.tfpnp_model import TFPnPModel
        model_params = TFPnPModel.default_parameters()
        model_params.sigma_min, model_params.sigma_max = sigma_range
        model_params.mu_min, model_params.mu_max = mu_range
        model = TFPnPModel(model_parameters=model_params,
                           geometry=experiment.experiment_params.geometry)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.policy.to(device).eval()
    elif 'policy_state_dict' in ckpt:
        policy = ResNetActor_ADMM(in_channels=5, n_action_steps=5,
                                  sigma_range=sigma_range,
                                  mu_range=mu_range).to(device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        return policy.eval()
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

def run_evaluation(test_images, test_labels, noise_levels, op, admm_step,
                   policy, device):
    all_results = {nl: {m: [] for m in METHODS} for nl in noise_levels}

    for noise_frac in noise_levels:
        print(f"\n{'='*60}\nNoise level: {noise_frac*100:.1f}%\n{'='*60}")

        for img_idx, gt in enumerate(test_images):
            sino_clean = op.forward(gt)
            SCALE = sino_clean.max() / gt.max()
            torch.manual_seed(img_idx * 100)
            sino_noisy = (sino_clean / SCALE
                          + noise_frac * (sino_clean / SCALE).std()
                          * torch.randn_like(sino_clean))
            y = sino_noisy * SCALE

            x_fbp = lion_fbp(y, op)
            x_fbp = ls_scale(gt, x_fbp).clamp(min=0)
            all_results[noise_frac]['FBP'].append(compute_metrics(gt, x_fbp))

            x_tv = tv_reconstruction(y, op, x0=x_fbp, lam=0.01, n_iters=200)
            x_tv = ls_scale(gt, x_tv).clamp(min=0)
            all_results[noise_frac]['TV'].append(compute_metrics(gt, x_tv))

            x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
            with torch.no_grad():
                for k in range(20):
                    x, z, u = admm_step(x, z, u, y, sigma=FIXED_SIGMA, mu=FIXED_MU)
            x_fix = ls_scale(gt, x).clamp(min=0)
            all_results[noise_frac]['Fixed PnP-ADMM'].append(compute_metrics(gt, x_fix))

            x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
            noise_level = torch.tensor([noise_frac], device=device)
            with torch.no_grad():
                for t in range(6):
                    iter_frac = torch.tensor([t / 6.0], device=device)
                    stop_logits, sigma_seq, mu_seq = policy(
                        to_4d(x), to_4d(z), to_4d(u), noise_level, iter_frac)
                    if t > 0 and F.softmax(stop_logits, -1)[0, 1].item() > 0.5:
                        break
                    for i in range(5):
                        x, z, u = admm_step(x, z, u, y,
                                            sigma=sigma_seq[0, i], mu=mu_seq[0, i])
            x_tfp = ls_scale(gt, x).clamp(min=0)
            all_results[noise_frac]['TFPnP'].append(compute_metrics(gt, x_tfp))

            print(f"  {test_labels[img_idx]}: "
                  f"FBP={all_results[noise_frac]['FBP'][-1]['psnr']:.1f} | "
                  f"TV={all_results[noise_frac]['TV'][-1]['psnr']:.1f} | "
                  f"Fixed={all_results[noise_frac]['Fixed PnP-ADMM'][-1]['psnr']:.1f} | "
                  f"TFPnP={all_results[noise_frac]['TFPnP'][-1]['psnr']:.1f}")

    return all_results


def save_csv(all_results, noise_levels, path):
    rows = []
    for nl in noise_levels:
        for method in METHODS:
            row = {'noise_level': f'{nl*100:.1f}%', 'method': method}
            for metric in METRIC_NAMES:
                if metric not in all_results[nl][method][0]:
                    continue
                vals = [r[metric] for r in all_results[nl][method]]
                row[f'{metric}_mean'] = np.mean(vals)
                row[f'{metric}_std'] = np.std(vals)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"Saved {path}")


def save_per_image_metrics(all_results, test_labels, test_indices, noise_levels, path):
    """Dump full per-image metrics for all methods/noise levels to JSON.

    Enables downstream analysis (reconstruction galleries, ranking, custom plots)
    without re-running the eval loop.

    Structure:
        {
          'test_labels': [...],
          'test_indices': [...],
          'noise_levels': [0.05, 0.075, 0.10],
          'methods': ['FBP', 'TV', 'Fixed PnP-ADMM', 'TFPnP'],
          'metrics': {
            '5.0%': {'FBP': {'psnr': [...], 'ssim': [...], 'haarpsi': [...]}, ...},
            '7.5%': {...},
            '10.0%': {...},
          }
        }
    """
    if hasattr(test_indices, 'tolist'):
        idx_list = test_indices.tolist()
    else:
        idx_list = list(test_indices)

    out = {
        'test_labels': list(test_labels),
        'test_indices': idx_list,
        'noise_levels': list(noise_levels),
        'methods': METHODS,
        'metrics': {}
    }
    for nl in noise_levels:
        nl_key = f'{nl*100:.1f}%'
        out['metrics'][nl_key] = {}
        for method in METHODS:
            method_metrics = {}
            for metric in METRIC_NAMES:
                if metric in all_results[nl][method][0]:
                    method_metrics[metric] = [
                        float(r[metric]) for r in all_results[nl][method]
                    ]
            out['metrics'][nl_key][method] = method_metrics

    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")


def plot_metrics_bars(all_results, n_test, noise_level, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    results = all_results[noise_level]
    for ax, metric, ylabel in zip(axes, METRIC_NAMES, METRIC_LABELS):
        if metric not in results[METHODS[0]][0]:
            continue
        means = [np.mean([r[metric] for r in results[m]]) for m in METHODS]
        bars = ax.bar(range(len(METHODS)), means, color=COLORS, alpha=0.9,
                      edgecolor='white', linewidth=1.5, width=0.65)
        bars[-1].set_edgecolor('#c0392b')
        bars[-1].set_linewidth(2.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHOD_LABELS, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
        fmt = '.2f' if metric == 'psnr' else '.3f'
        for i, m in enumerate(means):
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
            ax.text(i, m + offset, f'{m:{fmt}}', ha='center',
                    fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.08)
    plt.suptitle(f'Reconstruction Quality at {noise_level*100:.1f}% Noise '
                 f'— {n_test} Test Images',
                 fontsize=12, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_per_image_psnr(all_results, test_labels, noise_level, path, n_select=5):
    """Per-image PSNR for cases with smallest and largest TFPnP–Fixed PnP gap.

    Selects n_select images where TFPnP barely improves over Fixed PnP-ADMM
    (smallest gap — the floor of TFPnP's contribution) and n_select where TFPnP
    wins by the largest margin (largest gap — the ceiling). Group labels sit
    above the plot area via a blended transform, so they never overlap with
    bars regardless of bar heights.
    """
    from matplotlib.transforms import blended_transform_factory

    results = all_results[noise_level]
    n_test = len(test_labels)

    tfpnp_psnrs = np.array([r['psnr'] for r in results['TFPnP']])
    fixed_psnrs = np.array([r['psnr'] for r in results['Fixed PnP-ADMM']])
    gaps = tfpnp_psnrs - fixed_psnrs

    if n_test <= 2 * n_select:
        show_indices = np.argsort(gaps)
        smallest_count = n_test
        largest_count = 0
    else:
        sorted_indices = np.argsort(gaps)
        smallest = sorted_indices[:n_select]
        largest = sorted_indices[-n_select:]
        show_indices = np.concatenate([smallest, largest])
        smallest_count = n_select
        largest_count = n_select

    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(show_indices))
    width = 0.18
    offset = (len(METHODS) - 1) * width / 2

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        psnrs = [results[method][idx]['psnr'] for idx in show_indices]
        ax.bar(x_pos + i * width - offset, psnrs, width,
               label=method, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.5)

    gap_vals = [results['TFPnP'][idx]['psnr'] -
                results['Fixed PnP-ADMM'][idx]['psnr']
                for idx in show_indices]
    labels_subset = [
        f"{test_labels[idx].replace('test_', '#')}\nΔ={gap:+.2f}"
        for idx, gap in zip(show_indices, gap_vals)
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_subset, fontsize=8)

    if smallest_count > 0 and largest_count > 0:
        divider_x = smallest_count - 0.5
        ax.axvline(divider_x, color='gray', linestyle=':',
                   linewidth=1, alpha=0.6)

        # Group labels above the plot area, never overlap bars
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(smallest_count / 2 - 0.5, 1.02,
                'Smallest TFPnP–Fixed gap',
                transform=trans, ha='center', va='bottom',
                fontsize=9, color='#555', style='italic')
        ax.text(smallest_count + largest_count / 2 - 0.5, 1.02,
                'Largest TFPnP–Fixed gap',
                transform=trans, ha='center', va='bottom',
                fontsize=9, color='#555', style='italic')

        title = (f'Per-Image PSNR at {noise_level*100:.1f}% Noise — '
                 f'{smallest_count} smallest and {largest_count} largest '
                 f'TFPnP–Fixed PnP gaps')
    else:
        title = f'Per-Image PSNR at {noise_level*100:.1f}% Noise'

    ax.set_ylabel('PSNR (dB)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=30)
    ax.legend(fontsize=8, ncol=len(METHODS), loc='upper center',
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_metrics_vs_noise(all_results, noise_levels, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    noise_pcts = np.array([nl * 100 for nl in noise_levels])
    for ax, metric, ylabel in zip(axes, METRIC_NAMES, METRIC_LABELS):
        if metric not in all_results[noise_levels[0]][METHODS[0]][0]:
            continue
        for method, color in zip(METHODS, COLORS):
            means = [np.mean([r[metric] for r in all_results[nl][method]])
                     for nl in noise_levels]
            lw = 2.8 if method == 'TFPnP' else 1.8
            ms = 9 if method == 'TFPnP' else 6
            ax.plot(noise_pcts, means, marker='o', lw=lw, ms=ms,
                    color=color, label=method, alpha=0.95)
        ax.set_xlabel('Noise level (%)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(noise_pcts)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(METHODS),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.suptitle('Reconstruction Quality vs Noise Level',
                 fontsize=12, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--n_test_subset", type=int, default=0,
               help="Max test images. 0 = use all (389 from LION test split).")
    p.add_argument("--noise_levels", nargs="+", type=float,
                   default=[0.05, 0.075, 0.10])
    p.add_argument("--denoiser_path", type=str,
                   default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")

    output_dir = OUTPUT_BASE / args.experiment_name
    results_dir = RESULTS_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)

    test_images, test_indices = get_lion_split(
        split="test", geometry=geo, device=device)
    n_test_total = len(test_images)
    if args.n_test_subset > 0 and args.n_test_subset < n_test_total:
        test_images = test_images[:args.n_test_subset]
        test_indices = test_indices[:args.n_test_subset]
        print(f"Subset: {args.n_test_subset} of {n_test_total} test images")
    else:
        print(f"Using all {n_test_total} test images")
    test_labels = [f"test_{i}" for i in test_indices]

    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy = load_checkpoint(ckpt_dir, device)

    all_results = run_evaluation(test_images, test_labels, args.noise_levels,
                                  op, admm_step, policy, device)

    save_csv(all_results, args.noise_levels, results_dir / "all_results.csv")
    save_per_image_metrics(all_results, test_labels, test_indices,
                            args.noise_levels,
                            results_dir / "per_image_metrics.json")
    plot_metrics_bars(all_results, len(test_images), 0.05,
                      output_dir / "metrics_comparison.pdf")
    plot_per_image_psnr(all_results, test_labels, 0.05,
                        output_dir / "per_image_psnr.pdf")
    plot_metrics_vs_noise(all_results, args.noise_levels,
                          output_dir / "metrics_vs_noise.pdf")

    print(f"\nDone.")
    print(f"  Figures : {output_dir}/")
    print(f"  Metrics : {results_dir}/")


if __name__ == "__main__":
    main()