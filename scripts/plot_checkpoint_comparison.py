#!/usr/bin/env python3
"""
Plot a four-panel reconstruction comparison (GT / FBP / Fixed PnP-ADMM / TFPnP)
on a single validation image, using the best validation checkpoint.

Mirrors NB09 cell 27.

Usage:
    python scripts/plot_checkpoint_comparison.py --experiment_name run_02_pat_200
    python scripts/plot_checkpoint_comparison.py --experiment_name run_02_pat_200 --image_idx 0 --noise_std 0.05
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


# ── Output paths ───────────────────────────────────────────────────────
OUTPUT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")

# ── Baseline parameters (must match evaluate_run.py for consistency) ──
FIXED_SIGMA = 1.5
FIXED_MU = 20.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--image_idx", type=int, default=None,
               help="Specific val-image index. If unset, auto-picks a lung slice.")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--denoiser_path", type=str,
                   default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")

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

    # ── Load checkpoint ────────────────────────────────────────────────
    ckpt_dir = CHECKPOINT_BASE / args.experiment_name
    ckpt_path = ckpt_dir / "checkpoint_best_val.pth"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded: {ckpt_path.name}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    if 'val_psnr' in ckpt:
        print(f"  Val PSNR: {ckpt['val_psnr']:.2f} dB")

    # Reconstruct the policy with the same architecture as training
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
        policy_best = model.policy.to(device).eval()        # ← assign, not return
    elif 'policy_state_dict' in ckpt:
        policy_best = ResNetActor_ADMM(in_channels=5, n_action_steps=5,
                                       sigma_range=sigma_range,
                                       mu_range=mu_range).to(device)
        policy_best.load_state_dict(ckpt['policy_state_dict'])
        policy_best.eval()                                  # ← in-place, no return
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    # ── Pick a validation image ────────────────────────────────────────
    val_images, _ = get_lion_split(split="validation", geometry=geo, device=device)

    if args.image_idx is not None:
        if args.image_idx >= len(val_images):
            raise ValueError(f"image_idx {args.image_idx} >= len(val_images) {len(val_images)}")
        chosen_idx = args.image_idx
        print(f"Using val_images[{chosen_idx}] (user-specified)")
    else:
        lung_indices = [i for i, img in enumerate(val_images) if is_lung_slice(img)]
        if lung_indices:
            chosen_idx = lung_indices[len(lung_indices) // 2]
            print(f"Found {len(lung_indices)} lung slices; "
                f"auto-selected val_images[{chosen_idx}]")
        else:
            chosen_idx = 0
            print("No lung slices found; falling back to val_images[0]")

    test_gt = val_images[chosen_idx]
    print(f"Evaluating on val_images[{chosen_idx}] (held out from training)")

    # ── Forward project + noise ────────────────────────────────────────
    sino_clean = op.forward(test_gt)
    SCALE = sino_clean.max() / test_gt.max()
    torch.manual_seed(99)   # fixed seed → reproducible noise
    sino_noisy = (sino_clean / SCALE +
                  args.noise_std * (sino_clean / SCALE).std() *
                  torch.randn_like(sino_clean))
    y = sino_noisy * SCALE

    # ── FBP ────────────────────────────────────────────────────────────
    x_fbp = lion_fbp(y, op)
    x_fbp = ls_scale(test_gt, x_fbp).clamp(min=0)
    fbp_psnr = psnr(test_gt, x_fbp)

    # ── Fixed PnP-ADMM ─────────────────────────────────────────────────
    x_fix, z_fix, u_fix = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    with torch.no_grad():
        for k in range(20):
            x_fix, z_fix, u_fix = admm_step(x_fix, z_fix, u_fix, y,
                                             sigma=FIXED_SIGMA, mu=FIXED_MU)
    x_fix = ls_scale(test_gt, x_fix).clamp(min=0)
    fix_psnr = psnr(test_gt, x_fix)

    # ── TFPnP ──────────────────────────────────────────────────────────
    x, z, u = x_fbp.clone(), x_fbp.clone(), torch.zeros_like(x_fbp)
    noise_level = torch.tensor([args.noise_std], device=device)
    n_steps_run = 0

    print("\nTFPnP decisions:")
    with torch.no_grad():
        for t in range(6):
            iter_frac = torch.tensor([t / 6.0], device=device)
            stop_logits, sigma_seq, mu_seq = policy_best(
                to_4d(x), to_4d(z), to_4d(u), noise_level, iter_frac)
            stop_prob = F.softmax(stop_logits, dim=-1)[0, 1].item()
            if t > 0 and stop_prob > 0.5:
                print(f"  Stopped at decision {t} (P(stop)={stop_prob:.2f})")
                break
            for i in range(5):
                x, z, u = admm_step(x, z, u, y,
                                     sigma=sigma_seq[0, i], mu=mu_seq[0, i])
                n_steps_run += 1
            print(f"  Decision {t}: "
                  f"σ={[f'{v:.1f}' for v in sigma_seq[0].tolist()]}, "
                  f"μ={[f'{v:.1f}' for v in mu_seq[0].tolist()]}, "
                  f"PSNR={psnr(test_gt, x):.1f} dB")

    x_tfpnp = ls_scale(test_gt, x).clamp(min=0)
    tfpnp_psnr = psnr(test_gt, x_tfpnp)

    print(f"\nTotal ADMM steps: {n_steps_run}")
    print(f"FBP   : {fbp_psnr:.2f} dB")
    print(f"Fixed : {fix_psnr:.2f} dB  (σ={FIXED_SIGMA}, μ={FIXED_MU})")
    print(f"TFPnP : {tfpnp_psnr:.2f} dB")

    # ── Display ────────────────────────────────────────────────────────
    gt_disp = test_gt[0].cpu().numpy()
    vmin, vmax = gt_disp.min(), gt_disp.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(gt_disp, vmin=vmin, vmax=vmax, cmap='gray')
    axes[0].set_title("Ground truth (val image)")
    axes[0].axis("off")

    axes[1].imshow(x_fbp[0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='gray')
    axes[1].set_title(f"FBP\nPSNR={fbp_psnr:.1f} dB")
    axes[1].axis("off")

    axes[2].imshow(x_fix[0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='gray')
    axes[2].set_title(f"Fixed PnP-ADMM\n(σ={FIXED_SIGMA}, μ={FIXED_MU})\nPSNR={fix_psnr:.1f} dB")
    axes[2].axis("off")

    axes[3].imshow(x_tfpnp[0].cpu().numpy(), vmin=vmin, vmax=vmax, cmap='gray')
    axes[3].set_title(f"TFPnP (epoch {ckpt.get('epoch', '?')})\n"
                      f"{n_steps_run} steps\nPSNR={tfpnp_psnr:.1f} dB")
    axes[3].axis("off")

    plt.suptitle(f"Reconstruction Comparison — {args.experiment_name} "
             f"(val image {chosen_idx}, {args.noise_std*100:.1f}% noise)",
             y=1.02, fontsize=14)
    plt.tight_layout()

    out_path = output_dir / "checkpoint_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()