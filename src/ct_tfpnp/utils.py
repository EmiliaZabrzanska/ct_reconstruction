"""Project-wide small utilities."""

import json
import torch
from pathlib import Path
from ct_tfpnp.experiments.parallel_beam_ct import experiment
from ct_tfpnp.models.denoiser import DRUNetDenoiser
from ct_tfpnp.ct_ops.admm import ADMMStep
from LION.CTtools.ct_utils import make_operator


def to_4d(t: torch.Tensor, device=None) -> torch.Tensor:
    """
    Convert (1, H, W) → (1, 1, H, W) for network input.

    Networks (policy, critic) expect 4D (B, 1, H, W); LION operators
    and ADMMStep work in 3D (1, H, W). This helper inserts the batch
    dimension when needed, and optionally moves the tensor to a device.

    Idempotent: if `t` is already 4D, returns it unchanged.
    """
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0) if t.dim() == 3 else t


def read_metrics_config(ckpt_dir) -> dict:
    """
    Read the 'config' dict from a checkpoint directory's metrics_history.json.

    Returns an empty dict if the file doesn't exist or has no config key —
    callers should use .get(key, default) for safe field access.

    Used by all checkpoint-loading code to reconstruct training-time
    hyperparameters (σ/µ ranges, m, N, reward type, etc.).
    """
    ckpt_dir = Path(ckpt_dir)
    metrics_path = ckpt_dir / "metrics_history.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f).get('config', {})
    

def setup_admm(denoiser_path, device):
    """
    Standard ADMM environment setup used by all eval/plotting scripts.

    Builds: parallel-beam geometry, LION operator, frozen DRUNet denoiser,
    and the ADMMStep that ties them together. The denoiser is loaded from
    the given path and set to eval() with gradients disabled.

    Returns (geometry, operator, denoiser, admm_step).

    Imports are lazy to avoid circular dependencies — `utils` is imported
    by several modules and we don't want a load-time cycle through
    `ct_ops.admm` or `experiments.parallel_beam_ct`.
    """

    geo = experiment.experiment_params.geometry
    op = make_operator(geo)
    denoiser = DRUNetDenoiser(pretrained_path=denoiser_path).to(device)
    for p_ in denoiser.parameters():
        p_.requires_grad_(False)
    denoiser.eval()
    admm_step = ADMMStep(op=op, denoiser=denoiser, n_x_steps=6)
    return geo, op, denoiser, admm_step


def project_and_add_noise(gt, op, noise_std, seed):
    """
    Forward-project GT into a noisy sinogram, used by every eval/plot script.

    Uses LION's `op.forward(gt)` for the clean sinogram, normalises by the
    GT-to-sinogram scale, adds Gaussian noise at fractional level `noise_std`
    (relative to the scaled sinogram's std), and rescales back.

    `seed` controls the noise realisation, so different scripts using the
    same image at the same noise level get identical sinograms. By convention:
      - evaluate_run.py and plot_reconstruction_gallery.py use img_idx * 100
      - plot_checkpoint_comparison.py uses a fixed seed (typically 99)
    """
    sino_clean = op.forward(gt)
    SCALE = sino_clean.max() / gt.max()
    torch.manual_seed(seed)
    sino_noisy = (sino_clean / SCALE + noise_std * (sino_clean / SCALE).std()
                  * torch.randn_like(sino_clean))
    return sino_noisy * SCALE