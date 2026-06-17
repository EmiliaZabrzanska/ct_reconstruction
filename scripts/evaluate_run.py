#!/usr/bin/env python3
"""
Full evaluation script — produces all NB10 figures for a given experiment.

Usage:
    python -u scripts/evaluate_run.py --experiment_name run_04_pat_250_e80
    python -u scripts/evaluate_run.py --experiment_name run_04_pat_250_e80 \
        --fbpconvnet_ckpt results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth
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
from ct_tfpnp.models.fbpconvnet_image import FBPConvNetImage
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from LION.models.post_processing.FBPConvNet import FBPConvNet as _FBPConvNetParent

# ── Output paths (edit these once if your layout ever changes) ─────────
OUTPUT_BASE  = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
RESULTS_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/metrics")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")

# ── Constants ──────────────────────────────────────────────────────────
METHODS       = ['FBP', 'TV', 'DRUNet', 'FBPConvNet', 'Fixed PnP-ADMM', 'TFPnP']
METHOD_LABELS = ['FBP', 'TV', 'DRUNet\n(σ=10)', 'FBP-\nConvNet', 'Fixed\nPnP-ADMM', 'TFPnP\n(Ours)']
COLORS        = ['#bdc3c7', '#f39c12', '#9b59b6', '#1abc9c', '#3498db', '#e74c3c']
METRIC_NAMES = ['psnr', 'ssim', 'haarpsi']
METRIC_LABELS = ['PSNR (dB)', 'SSIM', 'HaarPSI']
FIXED_SIGMA = 1.5
FIXED_MU = 20.0
DRUNET_ALONE_SIGMA = 10.0


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


def load_fbpconvnet(ckpt_path, geo, device, sanity_image=None):
    """Load FBPConvNetImage from a checkpoint, with verification prints."""
    print(f"Loading FBPConvNet from {ckpt_path}")
    fbpc_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fbpc_params = _FBPConvNetParent.default_parameters()
    model = FBPConvNetImage(geometry_parameters=geo,
                            model_parameters=fbpc_params).to(device)
    load_result = model.load_state_dict(fbpc_ckpt['model_state_dict'])
    model.eval()

    print(f"  Path:        {ckpt_path}")
    print(f"  Epoch:       {fbpc_ckpt['epoch']}")
    print(f"  Val PSNR:    {fbpc_ckpt['val_psnr']:.2f} dB")
    if 'config' in fbpc_ckpt:
        print(f"  Config:      {fbpc_ckpt['config']}")
    print(f"  Load result: {load_result}")

    if sanity_image is not None:
        with torch.no_grad():
            test_out = model(to_4d(sanity_image))
        print(f"  Output range: [{test_out.min():.3f}, {test_out.max():.3f}] "
              f"(expect roughly [0, 2.5] for LION's µ scaling)")

    return model


def run_evaluation(test_images, test_labels, noise_levels, op, admm_step,
                   policy, denoiser, fbpconv_model, device):
    """If fbpconv_model is None, the FBPConvNet branch is skipped."""
    all_results = {nl: {m: [] for m in METHODS} for nl in noise_levels}

    for noise_frac in noise_levels:
        print(f"\n{'='*90}\nNoise level: {noise_frac*100:.1f}%\n{'='*90}")

        for img_idx, gt in enumerate(test_images):
            sino_clean = op.forward(gt)
            SCALE = sino_clean.max() / gt.max()
            torch.manual_seed(img_idx * 100)
            sino_noisy = (sino_clean / SCALE
                          + noise_frac * (sino_clean / SCALE).std()
                          * torch.randn_like(sino_clean))
            y = sino_noisy * SCALE

            # FBP
            x_fbp = lion_fbp(y, op)
            x_fbp = ls_scale(gt, x_fbp).clamp(min=0)
            all_results[noise_frac]['FBP'].append(compute_metrics(gt, x_fbp))

            # TV
            x_tv = tv_reconstruction(y, op, x0=x_fbp, lam=0.01, n_iters=200)
            x_tv = ls_scale(gt, x_tv).clamp(min=0)
            all_results[noise_frac]['TV'].append(compute_metrics(gt, x_tv))

            # DRUNet-alone (σ=10, no ADMM iterations)
            with torch.no_grad():
                x_drunet = denoiser(to_4d(x_fbp), DRUNET_ALONE_SIGMA).squeeze(0)
            x_drunet = ls_scale(gt, x_drunet).clamp(min=0)
            all_results[noise_frac]['DRUNet'].append(compute_metrics(gt, x_drunet))

            # FBPConvNet (only if checkpoint provided)
            if fbpconv_model is not None:
                with torch.no_grad():
                    x_fbpconv = fbpconv_model(to_4d(x_fbp))[0]
                x_fbpconv = ls_scale(gt, x_fbpconv).clamp(min=0)
                all_results[noise_frac]['FBPConvNet'].append(compute_metrics(gt, x_fbpconv))

            # Fixed PnP-ADMM (20 iterations at σ=1.5, μ=20)
            x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
            with torch.no_grad():
                for k in range(20):
                    x, z, u = admm_step(x, z, u, y, sigma=FIXED_SIGMA, mu=FIXED_MU)
            x_fix = ls_scale(gt, x).clamp(min=0)
            all_results[noise_frac]['Fixed PnP-ADMM'].append(compute_metrics(gt, x_fix))

            # TFPnP
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

            # Aligned per-image summary
            short = {'Fixed PnP-ADMM': 'Fixed', 'FBPConvNet': 'FBPConv'}
            parts = [f"{short.get(m, m)}={all_results[noise_frac][m][-1]['psnr']:>5.1f}"
                     for m in METHODS
                     if all_results[noise_frac][m]]
            print(f"  {test_labels[img_idx]:<8}: " + " | ".join(parts))

    return all_results


def print_summary_tables(all_results, noise_levels, noise_labels, has_haarpsi):
    """Aligned per-noise tables + compact PSNR-only paper-format table."""
    METHOD_W = 18
    PSNR_W = 13     # fits "XX.XX±X.XX"
    DECIMAL_W = 15  # fits "0.XXXX±0.XXXX"

    for noise_frac, noise_label in zip(noise_levels, noise_labels):
        print(f"\n{'='*60}")
        print(f"Noise level: {noise_label}")
        print(f"{'='*60}")
        print(f"{'Method':<{METHOD_W}} "
              f"{'PSNR (dB)':<{PSNR_W}} "
              f"{'SSIM':<{DECIMAL_W}} "
              f"{'HaarPSI':<{DECIMAL_W}}")
        print("-" * (METHOD_W + PSNR_W + 2 * DECIMAL_W + 3))

        for method in METHODS:
            psnrs = [r['psnr'] for r in all_results[noise_frac][method]]
            ssims = [r['ssim'] for r in all_results[noise_frac][method]]
            psnr_str = f"{np.mean(psnrs):5.2f}±{np.std(psnrs):.2f}"
            ssim_str = f"{np.mean(ssims):.4f}±{np.std(ssims):.4f}"
            hp_str = ""
            if has_haarpsi and 'haarpsi' in all_results[noise_frac][method][0]:
                hps = [r['haarpsi'] for r in all_results[noise_frac][method]]
                hp_str = f"{np.mean(hps):.4f}±{np.std(hps):.4f}"
            print(f"{method:<{METHOD_W}} "
                  f"{psnr_str:<{PSNR_W}} "
                  f"{ssim_str:<{DECIMAL_W}} "
                  f"{hp_str:<{DECIMAL_W}}")

    # Compact PSNR-only table
    NOISE_COL_W = 12
    print(f"\n{'='*60}")
    print("PSNR Summary (matching paper Table 5 format)")
    print(f"{'='*60}")
    print(f"{'Method':<{METHOD_W}}", end="")
    for label in noise_labels:
        print(f"  {label:>{NOISE_COL_W}}", end="")
    print()
    print("-" * (METHOD_W + len(noise_labels) * (NOISE_COL_W + 2)))
    for method in METHODS:
        print(f"{method:<{METHOD_W}}", end="")
        for nl in noise_levels:
            psnrs = [r['psnr'] for r in all_results[nl][method]]
            val_str = f"{np.mean(psnrs):5.2f}±{np.std(psnrs):.2f}"
            print(f"  {val_str:>{NOISE_COL_W}}", end="")
        print()

    # Paper reference table
    PAPER_NUM_W = 10
    print(f"\nPaper reference (Wei et al. 2022, Table 5):")
    print(f"{'Method':<{METHOD_W}}"
          f"  {'5%':>{PAPER_NUM_W}}"
          f"  {'7.5%':>{PAPER_NUM_W}}"
          f"  {'10%':>{PAPER_NUM_W}}"
          f"   {'Status in this reproduction'}")
    print("-" * (METHOD_W + 3 * (PAPER_NUM_W + 2) + 30))
    paper_rows = [
        ('FBP',           '18.37', '16.16', '14.32', '✓ implemented'),
        ('TV',            '21.47', '19.17', '16.95', '✓ implemented'),
        ('FBPconv',       '23.00', '22.94', '22.86', '✓ implemented (250×80 budget)'),
        ('RED-CNN',       '23.45', '23.11', '22.71', '✗ not in LION'),
        ('LPD',           '23.86', '23.44', '23.00', '— skipped per supervisor'),
        ('RPGD',          '24.29', '23.82', '23.38', '✗ not in LION'),
        ('TFPnP (paper)', '24.50', '24.23', '23.72', '✓ reproduced'),
    ]
    for name, v5, v75, v10, status in paper_rows:
        print(f"{name:<{METHOD_W}}"
              f"  {v5:>{PAPER_NUM_W}}"
              f"  {v75:>{PAPER_NUM_W}}"
              f"  {v10:>{PAPER_NUM_W}}"
              f"   {status}")
    print()
    print("Note: paper trains on natural images (PASCAL VOC); ours trains in-domain")
    print("on LIDC CT. Direct numerical comparison is therefore unfair — see Discussion.")


def save_csv(all_results, noise_levels, path):
    rows = []
    for nl in noise_levels:
        for method in METHODS:
            if not all_results[nl][method]:
                continue
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
    """Dump full per-image metrics for all methods/noise levels to JSON."""
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
            if not all_results[nl][method]:
                continue
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


def plot_per_image_psnr(all_results, test_labels, noise_level, path,
                        n_select=3, group_gap=0.5):
    """Per-image PSNR — TFPnP's worst losses and biggest wins vs next-best method.

    For each image, computes Δ = TFPnP − max(other methods). The most-negative
    Δs are TFPnP's biggest losses; the most-positive are its biggest wins.
    """
    from matplotlib.transforms import blended_transform_factory

    results = all_results[noise_level]
    n_test = len(test_labels)

    other_methods = [m for m in METHODS if m != 'TFPnP']
    tfpnp_psnrs = np.array([r['psnr'] for r in results['TFPnP']])
    other_psnrs = np.stack([
        np.array([r['psnr'] for r in results[m]]) for m in other_methods
    ], axis=0)
    next_best = other_psnrs.max(axis=0)
    delta = tfpnp_psnrs - next_best

    if n_test <= 2 * n_select:
        show_indices = np.argsort(delta)
        smallest_count = n_test
        largest_count = 0
    else:
        sorted_indices = np.argsort(delta)
        smallest = sorted_indices[:n_select]
        largest = sorted_indices[-n_select:]
        show_indices = np.concatenate([smallest, largest])
        smallest_count = n_select
        largest_count = n_select

    fig, ax = plt.subplots(figsize=(14, 6))

    x_pos = np.arange(len(show_indices), dtype=float)
    if smallest_count > 0 and largest_count > 0:
        x_pos[smallest_count:] += group_gap

    n_methods = len(METHODS)
    width = 0.90 / n_methods
    offset = (n_methods - 1) * width / 2

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        psnrs = [results[method][idx]['psnr'] for idx in show_indices]
        ax.bar(x_pos + i * width - offset, psnrs, width,
               label=method, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.5)

    labels_subset = [
        f"{test_labels[idx].replace('test_', '#')}\nΔ={delta[idx]:+.2f}"
        for idx in show_indices
    ]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_subset, fontsize=8)

    if smallest_count > 0 and largest_count > 0:
        divider_x = smallest_count - 0.5 + group_gap / 2
        ax.axvline(divider_x, color='gray', linestyle=':',
                   linewidth=1, alpha=0.6)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(smallest_count / 2 - 0.5, 1.02,
                'TFPnP outperformed by the most',
                transform=trans, ha='center', va='bottom',
                fontsize=9, color='#555', style='italic')
        ax.text(smallest_count + group_gap + largest_count / 2 - 0.5, 1.02,
                "TFPnP's biggest advantage",
                transform=trans, ha='center', va='bottom',
                fontsize=9, color='#555', style='italic')

        title = (f'Per-Image PSNR at {noise_level*100:.1f}% Noise — '
                 f"TFPnP's {smallest_count} largest losses and "
                 f"{largest_count} biggest wins vs. next-best method")
    else:
        title = f'Per-Image PSNR at {noise_level*100:.1f}% Noise'

    ax.set_ylabel('PSNR (dB)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=30)
    ax.legend(fontsize=8, ncol=n_methods, loc='upper center',
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
    p.add_argument("--fbpconvnet_ckpt", type=str, default=None,
                   help="Path to FBPConvNet checkpoint. If None, skip that baseline.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")

    # If no FBPConvNet checkpoint, prune it from the global method lists
    if not args.fbpconvnet_ckpt:
        global METHODS, METHOD_LABELS, COLORS
        idx = METHODS.index('FBPConvNet')
        METHODS = METHODS[:idx] + METHODS[idx+1:]
        METHOD_LABELS = METHOD_LABELS[:idx] + METHOD_LABELS[idx+1:]
        COLORS = COLORS[:idx] + COLORS[idx+1:]
        print("Note: --fbpconvnet_ckpt not provided — skipping FBPConvNet baseline")

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

    # Load FBPConvNet if checkpoint provided (with sanity check on test_images[0])
    fbpconv_model = None
    if args.fbpconvnet_ckpt:
        fbpconv_model = load_fbpconvnet(
            Path(args.fbpconvnet_ckpt), geo, device,
            sanity_image=test_images[0])

    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy = load_checkpoint(ckpt_dir, device)

    all_results = run_evaluation(test_images, test_labels, args.noise_levels,
                                  op, admm_step, policy, denoiser,
                                  fbpconv_model, device)

    # Pretty-print summary tables
    noise_labels = [f'{nl*100:.1f}%' for nl in args.noise_levels]
    has_haarpsi = 'haarpsi' in all_results[args.noise_levels[0]][METHODS[0]][0]
    print_summary_tables(all_results, args.noise_levels, noise_labels, has_haarpsi)

    # Save artefacts
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