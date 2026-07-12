#!/usr/bin/env python3
"""
Plot training curves from a saved metrics_history.json.

Usage:
    python scripts/plot_training_curves.py --experiment_name run_02_pat_200
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/figures")
CHECKPOINT_BASE = Path("/home/eaz21/rds/hpc-work/eaz21/results/learned")


def find_key(history, candidates):
    """
    Return the first key in `candidates` that exists with non-empty data.

    Args:
        history:    parsed metrics_history.json.
        candidates: key names to try, in priority order.

    Returns:
        The matching key, or None.
    """
    for c in candidates:
        if c in history and len(history[c]) > 0:
            return c
    return None


def plot_step_series(ax, values, title, color, zero_line=False):
    """
    Plot one per-update-step series.

    Args:
        ax:        matplotlib axis.
        values:    the series, or None to leave the panel empty but labelled.
        title:     panel title.
        color:     line colour.
        zero_line: draw a dashed line at y=0 (useful for the reward).
    """
    if values is not None:
        ax.plot(values, color=color, alpha=0.85)
        if zero_line:
            ax.axhline(0, color="#D62728", ls="--", lw=1)
    ax.set_xlabel("Update step")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    args = p.parse_args()

    # load metrics JSON
    metrics_path = CHECKPOINT_BASE / args.experiment_name / "metrics_history.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics_history.json at {metrics_path}. "
            f"Has training been run with metrics-saving enabled?")

    with open(metrics_path) as f:
        h = json.load(f)

    # prepare for different versions of keys (changed as project has evolved)
    k_critic = find_key(h, ['critic_loss', 'c_loss', 'loss_critic', 'loss'])
    k_reward = find_key(h, ['mean_reward', 'reward', 'reward_mean'])
    k_sigma  = find_key(h, ['mean_sigma', 'sigma', 'sigma_mean'])
    k_train  = find_key(h, ['epoch_psnr', 'train_psnr'])
    k_val    = find_key(h, ['val_psnr', 'validation_psnr'])

    print(f"Detected keys:")
    print(f"  critic  : {k_critic}")
    print(f"  reward  : {k_reward}")
    print(f"  sigma   : {k_sigma}")
    print(f"  train   : {k_train}")
    print(f"  val     : {k_val}")
    print(f"  All available: {[k for k in h if k != 'config']}")

    output_dir = OUTPUT_BASE / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

     # top row two RL signals, per update step
    plot_step_series(axes[0, 0], h.get(k_critic), "Critic loss (eq. 15)", "#5B9BD5")
    plot_step_series(axes[0, 1], h.get(k_reward), "Mean reward", "#F4B860", zero_line=True)

    # bottom left what the policy chooses
    plot_step_series(axes[1, 0], h.get(k_sigma), "Mean policy σ", "#70AD47")

    # bottom right qualit per epoch
    if k_train and k_val:
        epochs = list(range(1, len(h[k_train]) + 1))
        axes[1, 1].plot(epochs, h[k_train], "-o", color="#1F4E79", label="Train", lw=1.2, ms=4, alpha=0.75)
        axes[1, 1].plot(epochs, h[k_val], "-s", color="#C00000", label="Val", lw=1.2, ms=4, alpha=0.75)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("PSNR (dB)")
        axes[1, 1].set_title("Epoch mean PSNR")
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.2)

    plt.suptitle(f"TFPnP Training Curves — {args.experiment_name}", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = output_dir / "training_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out_path}")

    # ── Summary print ─────────────────────────────────────────────────
    if k_train and k_val:
        print(f"\nSummary:")
        print(f"  Epochs run    : {len(h[k_train])}")
        print(f"  Final train   : {h[k_train][-1]:.2f} dB")
        print(f"  Final val     : {h[k_val][-1]:.2f} dB")
        cfg = h.get('config', {})
        if 'best_val_psnr' in cfg:
            print(f"  Best val      : {cfg['best_val_psnr']:.2f} dB")


if __name__ == "__main__":
    main()