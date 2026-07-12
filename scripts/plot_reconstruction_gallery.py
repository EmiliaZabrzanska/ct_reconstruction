#!/usr/bin/env python
"""
Best / median / worst reconstruction gallery — every method plus ground truth.

Selects the best, median and worst test images by TFPnP's average rank across
PSNR / SSIM / HaarPSI, then reconstructs each with every method.

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

import matplotlib.pyplot as plt
import numpy as np
import torch

import ct_tfpnp
from ct_tfpnp.datasets.lidc import get_lion_split
from ct_tfpnp.utils import read_metrics_config, setup_admm

from evaluate_run import (
    CHECKPOINT_BASE, METRIC_NAMES, M_DEFAULT, N_DEFAULT, OUTPUT_BASE, RESULTS_BASE,
    compute_metrics, load_checkpoint, load_fbpconvnet, reconstruct_all,
)


def find_selections(per_image_data, noise_label):
    """
    Pick the best, median and worst test images by TFPnP's average rank.

    Args:
        per_image_data: parsed per_image_metrics.json.
        noise_label:    e.g. "5.0%" — must be a key in the JSON.

    Returns:
        ({'best': idx, 'median': idx, 'worst': idx}, list_of_metrics_used)
    """
    # load TFPnP metrics
    tfpnp_metrics = per_image_data['metrics'][noise_label]['TFPnP']
    available_metrics = [m for m in METRIC_NAMES if m in tfpnp_metrics]
    if not available_metrics:
        raise RuntimeError("No metrics available in per_image_metrics.json.")

    # find all metrics
    n_test = len(tfpnp_metrics[available_metrics[0]])

    # rank each individually
    ranks = np.zeros((n_test, len(available_metrics)))
    for j, metric in enumerate(available_metrics):
        order = np.argsort(np.array(tfpnp_metrics[metric]))[::-1]
        ranks_metric = np.empty(n_test)
        ranks_metric[order] = np.arange(1, n_test + 1)
        ranks[:, j] = ranks_metric

    # average acros rank
    avg_rank = ranks.mean(axis=1)
    sorted_by_avg = np.argsort(avg_rank)

    # select best, mid, worst
    selections = {
        'best':   int(sorted_by_avg[0]),
        'median': int(sorted_by_avg[len(sorted_by_avg) // 2]),
        'worst':  int(sorted_by_avg[-1]),
    }
    # print results
    print(f"Selected representative images (averaged rank across "
          f"{', '.join(m.upper() for m in available_metrics)}):\n")
    for kind, idx in selections.items():
        score_strs = [
            f"PSNR={tfpnp_metrics[m][idx]:.2f} dB" if m == 'psnr'
            else f"{m.upper()}={tfpnp_metrics[m][idx]:.3f}"
            for m in available_metrics
        ]
        print(f"  {kind.upper():<7} #{idx} (avg rank {avg_rank[idx]:.1f})  —  "
              + " | ".join(score_strs))

    return selections, available_metrics


def plot_gallery(selections, recons, available_metrics, gallery_methods,
                 noise_std, path):
    """
    3 x N grid: rows are best / median / worst, columns are Ground Truth plus each method.

    Args:
        selections:        {'best'|'median'|'worst': image index}.
        recons:            {image index: {method: reconstruction, 'metrics': {...}}}.
        available_metrics: metrics present, used to decide the per-cell caption.
        gallery_methods:   column order, starting with 'GT'.
        noise_std:         noise level, for the title.
        path:              output PDF path.
    """
    # check haarpsi availability and load columns
    has_haarpsi = 'haarpsi' in available_metrics
    n_cols = len(gallery_methods)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.7 * n_cols, 9.5), gridspec_kw={'wspace': 0.04, 'hspace': 0.15})

    # plot each row (best, median, worst)
    for row_idx, (kind, img_idx) in enumerate(selections.items()):
        r = recons[img_idx]
        gt_np = r['GT'][0].cpu().numpy()
        vmin, vmax = gt_np.min(), gt_np.max()

        # plot each column (GT + methods)
        for col_idx, method in enumerate(gallery_methods):
            ax = axes[row_idx, col_idx]
            ax.imshow(r[method][0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(method, fontsize=10)
            if method != 'GT':
                m = r['metrics'][method]
                parts = [f"{m['psnr']:.2f}", f"{m['ssim']:.3f}"]
                if has_haarpsi:
                    parts.append(f"{m['haarpsi']:.3f}")
                ax.set_xlabel(' / '.join(parts), fontsize=8, labelpad=4)

    plt.suptitle(f"Reconstruction gallery — best / median / worst at {noise_std*100:.1f}% noise", fontsize=13, y=0.95)
    fig.text(0.5, 0.08, "Per-cell labels: PSNR (dB) / SSIM" + (" / HaarPSI" if has_haarpsi else ""), ha='center', fontsize=9, style='italic', color='#555')

    plt.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--noise_std", type=float, default=0.05, help="Noise level for gallery (default 0.05). Must exist in JSON.")
    p.add_argument("--denoiser_path", type=str, default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    p.add_argument("--fbpconvnet_ckpt", type=str, default=None, help="Optional FBPConvNet checkpoint.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")
    print(f"noise      : {args.noise_std*100:.1f}%")

    # load metrics JSON
    json_path = RESULTS_BASE / args.experiment_name / "per_image_metrics.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Need {json_path}. Run evaluate_run.py first to generate it.")
    
    with open(json_path) as f:
        per_image_data = json.load(f)

    noise_label = f"{args.noise_std*100:.1f}%"
    
    if noise_label not in per_image_data['metrics']:
        
        raise ValueError(f"Noise level {noise_label} not in JSON. "
                         f"Available: {list(per_image_data['metrics'].keys())}")

    # pick images
    selections, available_metrics = find_selections(per_image_data, noise_label)

    # setup geometry, operator, denoiser, and ADMM step
    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)

    # load policy
    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy = load_checkpoint(ckpt_dir, device)

    # load training config
    cfg = read_metrics_config(ckpt_dir)
    m = int(cfg.get('m', M_DEFAULT))
    N = int(cfg.get('N', N_DEFAULT))

    # load images
    test_images = get_lion_split(split="test", geometry=geo, device=device)

    # load FBPConvNet 
    fbpconv_model = None
    if args.fbpconvnet_ckpt:
        fbpconv_model = load_fbpconvnet(Path(args.fbpconvnet_ckpt), geo, device, sanity_image=test_images[0])

    # build gallery methods list
    gallery_methods = ['GT', 'FBP', 'TV', 'DRUNet']
    if fbpconv_model is not None:
        gallery_methods.append('FBPConvNet')
    gallery_methods += ['Fixed PnP-ADMM', 'TFPnP']

    # reconstruct selected images
    print(f"\nReconstructing 3 images (best / median / worst) with {len(gallery_methods) - 1} methods + GT ...")
    recons = {}

    for k, (kind, img_idx) in enumerate(selections.items()):
        
        gt = test_images[img_idx]
        r = reconstruct_all(gt, img_idx, args.noise_std, op, admm_step, policy, denoiser, fbpconv_model, device, m=m, N=N)
        r['GT'] = gt
        r['metrics'] = {method: compute_metrics(gt, r[method]) for method in gallery_methods if method != 'GT'}
        recons[img_idx] = r
        print(f"  [{k+1}/3] {kind} (test_{img_idx}) reconstructed")

    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_gallery(selections, recons, available_metrics, gallery_methods, args.noise_std, output_dir / "gallery.pdf")

    print(f"\nDone. Gallery in {output_dir}/")


if __name__ == "__main__":
    main()