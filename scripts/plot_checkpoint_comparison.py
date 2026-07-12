#!/usr/bin/env python3
"""
Reconstruction comparison panel on a single validation image, using the best
validation checkpoint. Shows GT plus every baseline (FBP, TV, DRUNet, FBPConvNet,
Fixed PnP-ADMM, TFPnP). FBPConvNet is optional.

Usage:
    python -u scripts/plot_checkpoint_comparison.py --experiment_name run_04_pat_250_e80
    python -u scripts/plot_checkpoint_comparison.py --experiment_name run_04_pat_250_e80 \
        --fbpconvnet_ckpt results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import matplotlib.pyplot as plt

import ct_tfpnp
from ct_tfpnp.ct_ops.tv import tv_reconstruction
from ct_tfpnp.datasets.lidc import get_lion_split, is_lung_slice
from ct_tfpnp.evaluation.metrics import ls_scale, psnr_np as psnr
from ct_tfpnp.training.environment import run_episode
from ct_tfpnp.utils import read_metrics_config, setup_admm, to_4d

from evaluate_run import (
    CHECKPOINT_BASE, 
    DRUNET_ALONE_SIGMA, 
    FIXED_MU, 
    FIXED_PNP_ITERS, 
    FIXED_SIGMA,
    M_DEFAULT, 
    N_DEFAULT, 
    OUTPUT_BASE, 
    TV_LAM, 
    TV_N_ITERS,
    load_checkpoint, 
    load_fbpconvnet, 
    simulate,
)

# set consistent seedd
COMPARISON_SEED = 99


def pick_image(val_images, image_idx=None):
    """
    Choose which validation image to display.

    Args:
        val_images: list of ground-truth tensors.
        image_idx:  explicit index, or None to auto-pick.

    Returns:
        The chosen index.
    """
    # use user specified index if given
    if image_idx is not None:
        if image_idx >= len(val_images):
            raise ValueError(f"image_idx {image_idx} >= len(val_images) {len(val_images)}")
        print(f"Using val_images[{image_idx}] (user-specified)")
        return image_idx

    # select lung slice if available
    lung_indices = [i for i, img in enumerate(val_images) if is_lung_slice(img)]
    if lung_indices:
        chosen = lung_indices[len(lung_indices) // 2]
        print(f"Found {len(lung_indices)} lung slices; auto-selected val_images[{chosen}]")
        return chosen

    # use first slice if no lungs found
    print("No lung slices found; falling back to val_images[0]")
    return 0

def main():
    # set cli args
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--image_idx", type=int, default=None, help="Specific val-image index. If unset, auto-picks a lung slice.")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--denoiser_path", type=str, default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    p.add_argument("--fbpconvnet_ckpt", type=str, default=None, help="Optional FBPConvNet checkpoint. Skipped if not provided.")
    args = p.parse_args()

    # set device and print info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")

    # set output directory
    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # define geometry, operator, denoiser, and admm step
    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)

    # load trained policy from checkpoint
    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    policy = load_checkpoint(ckpt_dir, device)

    # m and N as used at training time
    cfg = read_metrics_config(ckpt_dir)
    m = int(cfg.get('m', M_DEFAULT))
    N = int(cfg.get('N', N_DEFAULT))

    # pick validation image
    val_images = get_lion_split(split="validation", geometry=geo, device=device)
    chosen_idx = pick_image(val_images, args.image_idx)
    test_gt = val_images[chosen_idx]
    print(f"Evaluating on val_images[{chosen_idx}] (held out from training)")

    # load FBPConvNet model
    fbpconv_model = None
    if args.fbpconvnet_ckpt:
        fbpconv_model = load_fbpconvnet(Path(args.fbpconvnet_ckpt), geo, device, sanity_image=test_gt)

    # compute noisy sinogram and FBP reconstruction
    y, x_fbp = simulate(test_gt, op, args.noise_std, seed=COMPARISON_SEED)

    # scale FBP to match GT and compute PSNR
    x_fbp_s = ls_scale(test_gt, x_fbp).clamp(min=0)
    fbp_psnr = psnr(test_gt, x_fbp_s)

    # compute TV reconstruction and PSNR
    x_tv = ls_scale(test_gt, tv_reconstruction(y, op, x0=x_fbp, lam=TV_LAM, n_iters=TV_N_ITERS)).clamp(min=0)
    tv_psnr = psnr(test_gt, x_tv)

    # compute DRUNet-alone reconstruction and PSNR
    with torch.no_grad():
        x_drunet = denoiser(to_4d(x_fbp), DRUNET_ALONE_SIGMA).squeeze(0)
    x_drunet = ls_scale(test_gt, x_drunet).clamp(min=0)
    drunet_psnr = psnr(test_gt, x_drunet)

    # compute FBPConvNet reconstruction and PSNR if model is provided
    fbpconv_psnr = None
    x_fbpconv = None
    if fbpconv_model is not None:
        with torch.no_grad():
            x_fbpconv = fbpconv_model(to_4d(x_fbp))[0]
        x_fbpconv = ls_scale(test_gt, x_fbpconv).clamp(min=0)
        fbpconv_psnr = psnr(test_gt, x_fbpconv)

    # compute Fixed PnP-ADMM reconstruction and PSNR
    x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    with torch.no_grad():
        for _ in range(FIXED_PNP_ITERS):
            x, z, u = admm_step(x, z, u, y, sigma=FIXED_SIGMA, mu=FIXED_MU)
    x_fix = ls_scale(test_gt, x).clamp(min=0)
    fix_psnr = psnr(test_gt, x_fix)

    # compute TFPnP reconstruction and PSNR
    ep = run_episode(policy, admm_step, test_gt, op, noise_std=args.noise_std, m=m, N=N, stochastic=False, seed=COMPARISON_SEED)
    x_tfpnp = ls_scale(test_gt, ep['x']).clamp(min=0)
    tfpnp_psnr = psnr(test_gt, x_tfpnp)

    # print results
    print("\nTFPnP decisions:")
    for d in range(ep['n_decisions']):
        sig = ep['sigmas'][d * m:(d + 1) * m]
        mu = ep['mus'][d * m:(d + 1) * m]
        print(f"  Decision {d}: σ={[f'{v:.1f}' for v in sig]}, μ={[f'{v:.1f}' for v in mu]}")
    print(f"  Stopped after {ep['n_decisions']} decisions ({ep['n_admm_steps']} ADMM steps)")

    print(f"\nFBP  : {fbp_psnr:.2f} dB")
    print(f"TV     : {tv_psnr:.2f} dB  (λ={TV_LAM})")
    print(f"DRUNet : {drunet_psnr:.2f} dB  (σ={DRUNET_ALONE_SIGMA})")

    if fbpconv_psnr is not None:
        print(f"FBPConvNet : {fbpconv_psnr:.2f} dB")

    print(f"Fixed PnP : {fix_psnr:.2f} dB  (σ={FIXED_SIGMA}, μ={FIXED_MU})")
    print(f"TFPnP     : {tfpnp_psnr:.2f} dB")

    # plot comparison panels
    gt_disp = test_gt[0].cpu().numpy()
    vmin, vmax = gt_disp.min(), gt_disp.max()

    panels = [
        (gt_disp, "Ground truth"),
        (x_fbp[0].cpu().numpy(), f"FBP\nPSNR={fbp_psnr:.1f} dB"),
        (x_tv[0].cpu().numpy(), f"TV (λ=0.01)\nPSNR={tv_psnr:.1f} dB"),
        (x_drunet[0].cpu().numpy(), f"DRUNet-alone (σ={DRUNET_ALONE_SIGMA})\nPSNR={drunet_psnr:.1f} dB"),
        (x_fix[0].cpu().numpy(), f"Fixed PnP-ADMM (σ={FIXED_SIGMA}, μ={FIXED_MU})\nPSNR={fix_psnr:.1f} dB"),
        (x_tfpnp[0].cpu().numpy(), f"TFPnP\n{ep['n_admm_steps']} steps\nPSNR={tfpnp_psnr:.1f} dB"),
    ]
    if fbpconv_model is not None:
        panels.insert(-1, (x_fbpconv[0].cpu().numpy(), f"FBPConvNet (250×80)\nPSNR={fbpconv_psnr:.1f} dB"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Reconstruction Comparison — {args.experiment_name} (val image {chosen_idx}, {args.noise_std*100:.1f}% noise)", y=1.02, fontsize=14)
    plt.tight_layout()

    out_path = output_dir / "checkpoint_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()