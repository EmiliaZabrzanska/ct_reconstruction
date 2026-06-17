#!/usr/bin/env python3
"""
Reconstruction comparison panel on a single validation image, using the best
validation checkpoint. Shows GT + all baselines (FBP, TV, DRUNet, FBPConvNet,
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
import torch.nn.functional as F
import matplotlib.pyplot as plt

import ct_tfpnp
from ct_tfpnp.models.policy import ResNetActor_ADMM
from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.ct_ops.tv import tv_reconstruction
from ct_tfpnp.evaluation.metrics import psnr_np as psnr, ls_scale
from ct_tfpnp.datasets.lidc import get_lion_split, is_lung_slice
from ct_tfpnp.utils import to_4d, read_metrics_config, setup_admm

from evaluate_run import load_fbpconvnet, FIXED_SIGMA, FIXED_MU, DRUNET_ALONE_SIGMA

OUTPUT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--image_idx", type=int, default=None,
                   help="Specific val-image index. If unset, auto-picks a lung slice.")
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--denoiser_path", type=str,
                   default="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth")
    p.add_argument("--fbpconvnet_ckpt", type=str, default=None,
                   help="Optional FBPConvNet checkpoint. Skipped if not provided.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ct_tfpnp   : {ct_tfpnp.__version__}")
    print(f"device     : {device}")
    print(f"experiment : {args.experiment_name}")

    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────
    geo, op, denoiser, admm_step = setup_admm(args.denoiser_path, device)

    # ── Load TFPnP checkpoint ──────────────────────────────────────────
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
        model = TFPnPModel(model_parameters=model_params, geometry=geo)
        model.load_state_dict(ckpt['model_state_dict'])
        policy_best = model.policy.to(device).eval()
    elif 'policy_state_dict' in ckpt:
        policy_best = ResNetActor_ADMM(in_channels=5, n_action_steps=5,
                                       sigma_range=sigma_range,
                                       mu_range=mu_range).to(device)
        policy_best.load_state_dict(ckpt['policy_state_dict'])
        policy_best.eval()
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

    # ── Load FBPConvNet (optional) ─────────────────────────────────────
    fbpconv_model = None
    if args.fbpconvnet_ckpt:
        fbpconv_model = load_fbpconvnet(
            Path(args.fbpconvnet_ckpt), geo, device,
            sanity_image=test_gt)

    # ── Forward project + noise ────────────────────────────────────────
    sino_clean = op.forward(test_gt)
    SCALE = sino_clean.max() / test_gt.max()
    torch.manual_seed(99)
    sino_noisy = (sino_clean / SCALE
                  + args.noise_std * (sino_clean / SCALE).std()
                  * torch.randn_like(sino_clean))
    y = sino_noisy * SCALE

    # ── FBP ────────────────────────────────────────────────────────────
    x_fbp = lion_fbp(y, op)
    x_fbp = ls_scale(test_gt, x_fbp).clamp(min=0)
    fbp_psnr = psnr(test_gt, x_fbp)

    # ── TV ─────────────────────────────────────────────────────────────
    x_tv = tv_reconstruction(y, op, x0=x_fbp, lam=0.01, n_iters=200)
    x_tv = ls_scale(test_gt, x_tv).clamp(min=0)
    tv_psnr = psnr(test_gt, x_tv)

    # ── DRUNet-alone ───────────────────────────────────────────────────
    with torch.no_grad():
        x_drunet = denoiser(to_4d(x_fbp), DRUNET_ALONE_SIGMA).squeeze(0)
    x_drunet = ls_scale(test_gt, x_drunet).clamp(min=0)
    drunet_psnr = psnr(test_gt, x_drunet)

    # ── FBPConvNet (if loaded) ─────────────────────────────────────────
    fbpconv_psnr = None
    x_fbpconv = None
    if fbpconv_model is not None:
        with torch.no_grad():
            x_fbpconv = fbpconv_model(to_4d(x_fbp))[0]
        x_fbpconv = ls_scale(test_gt, x_fbpconv).clamp(min=0)
        fbpconv_psnr = psnr(test_gt, x_fbpconv)

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
    print(f"FBP        : {fbp_psnr:.2f} dB")
    print(f"TV         : {tv_psnr:.2f} dB  (λ=0.01)")
    print(f"DRUNet     : {drunet_psnr:.2f} dB  (σ={DRUNET_ALONE_SIGMA})")
    if fbpconv_psnr is not None:
        print(f"FBPConvNet : {fbpconv_psnr:.2f} dB")
    print(f"Fixed PnP  : {fix_psnr:.2f} dB  (σ={FIXED_SIGMA}, μ={FIXED_MU})")
    print(f"TFPnP      : {tfpnp_psnr:.2f} dB")

    # ── Display ────────────────────────────────────────────────────────
    gt_disp = test_gt[0].cpu().numpy()
    vmin, vmax = gt_disp.min(), gt_disp.max()

    panels = [
        (gt_disp, "Ground truth"),
        (x_fbp[0].cpu().numpy(), f"FBP\nPSNR={fbp_psnr:.1f} dB"),
        (x_tv[0].cpu().numpy(), f"TV (λ=0.01)\nPSNR={tv_psnr:.1f} dB"),
        (x_drunet[0].cpu().numpy(),
         f"DRUNet-alone (σ={DRUNET_ALONE_SIGMA})\nPSNR={drunet_psnr:.1f} dB"),
        (x_fix[0].cpu().numpy(),
         f"Fixed PnP-ADMM\n(σ={FIXED_SIGMA}, μ={FIXED_MU})\nPSNR={fix_psnr:.1f} dB"),
        (x_tfpnp[0].cpu().numpy(),
         f"TFPnP (epoch {ckpt.get('epoch', '?')})\n"
         f"{n_steps_run} steps\nPSNR={tfpnp_psnr:.1f} dB"),
    ]
    if fbpconv_model is not None:
        # Insert FBPConvNet between Fixed PnP and TFPnP
        panels.insert(-1,
            (x_fbpconv[0].cpu().numpy(),
             f"FBPConvNet (250×80)\nPSNR={fbpconv_psnr:.1f} dB"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis("off")

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