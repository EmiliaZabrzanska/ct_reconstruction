# scripts

Thin wrappers around the `ct_tfpnp` package, run from the repo root with the `mphil_ct` env active.

## Setup

| Script | Purpose |
|---|---|
| `download_denoiser.py` | Fetch `drunet_gray.pth` into `results/baselines/`. |

## Training

| Script | Purpose |
|---|---|
| `train_tfpnp.py` | Train the TFPnP policy (the main experiment). |
| `train_fbpconvnet.py` | FBPConvNet post-processing baseline. |
| `train_drunet_ct.py` | Optional CT-domain DRUNet denoiser. |

> Batch sizes must be > 1 due to the policy using BatchNorm.

## Evaluation and figures

Run `evaluate_run.py` first as it writes `per_image_metrics.json`, which is used by `plot_reconstruction_gallery.py`.

| Script | Produces |
|---|---|
| `evaluate_run.py` | Metrics CSV, per-image JSON, metric figures. |
| `plot_training_curves.py` | Training curves |
| `plot_checkpoint_comparison.py` | Panel showing one image across all methods. |
| `plot_reconstruction_gallery.py` | Best / median / worst gallery. |
| `plot_policy_behaviour.py` | Parameter trajectories. |