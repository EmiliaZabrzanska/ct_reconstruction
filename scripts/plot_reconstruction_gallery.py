#!/usr/bin/env python3
"""
Best / median / worst reconstruction gallery — all 6 methods + GT.

Selects the best, median, and worst test images (averaged rank across PSNR /
SSIM / HaarPSI of TFPnP) and reconstructs each with FBP, TV, DRUNet,
FBPConvNet (optional), Fixed PnP-ADMM, and TFPnP.

Requires per_image_metrics.json from evaluate_run.py.

Usage:
    python -u scripts/plot_reconstruction_gallery.py --experiment_name run_04_pat_250_e80
    python -u scripts/plot_reconstruction_gallery.py --experiment_name run_04_pat_250_e80 \
        --fbpconvnet_ckpt results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from evaluate_run import (
    compute_metrics, load_checkpoint, load_fbpconvnet,
    FIXED_SIGMA, FIXED_MU, DRUNET_ALONE_SIGMA,
    OUTPUT_BASE, RESULTS_BASE, CHECKPOINT_BASE,
)

import ct_tfpnp
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.ct_ops.tv import tv_reconstruction
from ct_tfpnp.evaluation.metrics import ls_scale
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.utils import to_4d, setup_admm


METRIC_NAMES = ['psnr', 'ssim', 'haarpsi']


def find_selections(per_image_data, noise_label):
    """Pick best/median/worst test images by average rank across all available metrics."""
    tfpnp_metrics = per_image_data['metrics'][noise_label]['TFPnP']
    available_metrics = [m for m in METRIC_NAMES if m in tfpnp_metrics]
    if not available_metrics:
        raise RuntimeError("No metrics available in per_image_metrics.json.")

    n_test = len(tfpnp_metrics[available_metrics[0]])
    ranks = np.zeros((n_test, len(available_metrics)))
    for j, metric in enumerate(available_metrics):
        scores = np.array(tfpnp_metrics[metric])
        order = np.argsort(scores)[::-1]
        ranks_metric = np.empty(n_test)
        ranks_metric[order] = np.arange(1, n_test + 1)
        ranks[:, j] = ranks_metric

    avg_rank = ranks.mean(axis=1)
    sorted_by_avg = np.argsort(avg_rank)

    selections = {
        'best':   int(sorted_by_avg[0]),
        'median': int(sorted_by_avg[len(sorted_by_avg) // 2]),
        'worst':  int(sorted_by_avg[-1]),
    }

    print(f"Selected representative images (averaged rank across "
          f"{', '.join(m.upper() for m in available_metrics)}):\n")
    for kind, idx in selections.items():
        score_strs = []
        for m in available_metrics:
            score = tfpnp_metrics[m][idx]
            if m == 'psnr':
                score_strs.append(f"PSNR={score:.2f} dB")
            else:
                score_strs.append(f"{m.upper()}={score:.3f}")
        print(f"  {kind.upper():<7} #{idx} (avg rank {avg_rank[idx]:.1f})  —  "
              + " | ".join(score_strs))

    return selections, available_metrics


def reconstruct_for_gallery(gt, noise_std, img_idx, op, admm_step,
                             policy, denoiser, fbpconv_model, device):
    """Run all 6 methods on one image, returning images + metrics dict."""
    sino_clean = op.forward(gt)
    SCALE = sino_clean.max() / gt.max()
    torch.manual_seed(img_idx * 100)
    sino_noisy = (sino_clean / SCALE +
                  noise_std * (sino_clean / SCALE).std() *
                  torch.randn_like(sino_clean))
    y = sino_noisy * SCALE

    # FBP
    x_fbp = lion_fbp(y, op)
    x_fbp = ls_scale(gt, x_fbp).clamp(min=0)

    # TV
    x_tv_raw = tv_reconstruction(y, op, x0=x_fbp, lam=0.01, n_iters=200)
    x_tv = ls_scale(gt, x_tv_raw).clamp(min=0)

    # DRUNet-alone
    with torch.no_grad():
        x_drunet_raw = denoiser(to_4d(x_fbp), DRUNET_ALONE_SIGMA).squeeze(0)
    x_drunet = ls_scale(gt, x_drunet_raw).clamp(min=0)

    # FBPConvNet (optional)
    x_fbpconv = None
    if fbpconv_model is not None:
        with torch.no_grad():
            x_fbpconv_raw = fbpconv_model(to_4d(x_fbp))[0]
        x_fbpconv = ls_scale(gt, x_fbpconv_raw).clamp(min=0)

    # Fixed PnP-ADMM
    x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    with torch.no_grad():
        for k in range(20):
            x, z, u = admm_step(x, z, u, y, sigma=FIXED_SIGMA, mu=FIXED_MU)
    x_fix = ls_scale(gt, x).clamp(min=0)

    # TFPnP
    x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    noise_level = torch.tensor([noise_std], device=device)
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

    recon = {
        'GT':              gt,
        'FBP':             x_fbp,
        'TV':              x_tv,
        'DRUNet':          x_drunet,
        'Fixed PnP-ADMM':  x_fix,
        'TFPnP':           x_tfp,
        'metrics': {
            'FBP':             compute_metrics(gt, x_fbp),
            'TV':              compute_metrics(gt, x_tv),
            'DRUNet':          compute_metrics(gt, x_drunet),
            'Fixed PnP-ADMM':  compute_metrics(gt, x_fix),
            'TFPnP':           compute_metrics(gt, x_tfp),
        }
    }
    if x_fbpconv is not None:
        recon['FBPConvNet'] = x_fbpconv
        recon['metrics']['FBPConvNet'] = compute_metrics(gt, x_fbpconv)

    return recon


def plot_gallery(selections, recons, available_metrics, gallery_methods,
                 noise_std, path):
    """3 × N grid: rows = best/median/worst, cols = GT + methods."""
    has_haarpsi = 'haarpsi' in available_metrics
    n_cols = len(gallery_methods)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.7 * n_cols, 9.5),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.15})

    for row_idx, (kind, img_idx) in enumerate(selections.items()):
        r = recons[img_idx]
        gt_np = r['GT'][0].cpu().numpy()
        vmin, vmax = gt_np.min(), gt_np.max()

        for col_idx, method in enumerate(gallery_methods):
            ax = axes[row_idx, col_idx]
            ax.imshow(r[method][0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(method, fontsize=10)
            if method != 'GT':
                m = r['metrics'][method]
                parts = [f"{m['psnr']:.2f}", f"{m['ssim']:.3f}"]
                if has_haarpsi:
                    parts.append(f"{m['haarpsi']:.3f}")
                ax.set_xlabel(' / '.join(parts), fontsize=8, labelpad=4)

    plt.suptitle(
        f"Reconstruction gallery — best / median / worst at {noise_std*100:.1f}% noise",
        fontsize=13, y=0.95)

    fig.text(0.5, 0.08,
             "Per-cell labels: PSNR (dB) / SSIM" + (" / HaarPSI" if has_haarpsi else ""),
             ha='center', fontsize=9, style='italic', color='#555')

    plt.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--noise_std", type=float, default=0.05,
                   help="Noise level for gallery (default 0.05). Must exist in JSON.")
    p.add_argument("--denoiser_path", type=str,
                   default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    p.add_argument("--fbpconvnet_ckpt", type=str, default=None,
                   help="Optional FBPConvNet checkpoint.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")
    print(f"noise      : {args.noise_std*100:.1f}%")

    # ── Load per-image metrics JSON ─────────────────────────────────────
    json_path = RESULTS_BASE / args.experiment_name / "per_image_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Need {json_path}. Run evaluate_run.py first to generate it.")
    with open(json_path) as f:
        per_image_data = json.load(f)

    noise_label = f"{args.noise_std*100:.1f}%"
    if noise_label not in per_image_data['metrics']:
        raise ValueError(
            f"Noise level {noise_label} not in JSON. "
            f"Available: {list(per_image_data['metrics'].keys())}")

    # ── Pick representative images ──────────────────────────────────────
    selections, available_metrics = find_selections(per_image_data, noise_label)

    # ── Setup and load models ───────────────────────────────────────────
    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)
    policy = load_checkpoint(CHECKPOINT_BASE / args.experiment_name, device)

    test_images, _ = get_lion_split(split="test", geometry=geo, device=device)

    fbpconv_model = None
    if args.fbpconvnet_ckpt:
        fbpconv_model = load_fbpconvnet(
            Path(args.fbpconvnet_ckpt), geo, device,
            sanity_image=test_images[0])

    # ── Build the gallery_methods list (dynamic on FBPConvNet) ─────────
    gallery_methods = ['GT', 'FBP', 'TV', 'DRUNet']
    if fbpconv_model is not None:
        gallery_methods.append('FBPConvNet')
    gallery_methods += ['Fixed PnP-ADMM', 'TFPnP']

    # ── Reconstruct ─────────────────────────────────────────────────────
    print(f"\nReconstructing 3 images (best / median / worst) with "
          f"{len(gallery_methods) - 1} methods + GT ...")
    recons = {}
    for k, (kind, img_idx) in enumerate(selections.items()):
        recons[img_idx] = reconstruct_for_gallery(
            test_images[img_idx], args.noise_std, img_idx,
            op, admm_step, policy, denoiser, fbpconv_model, device)
        print(f"  [{k+1}/3] {kind} (test_{img_idx}) reconstructed")

    # ── Plot ────────────────────────────────────────────────────────────
    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_gallery(selections, recons, available_metrics, gallery_methods,
                 args.noise_std, output_dir / "gallery.pdf")

    print(f"\nDone. Gallery in {output_dir}/")


if __name__ == "__main__":
    main()