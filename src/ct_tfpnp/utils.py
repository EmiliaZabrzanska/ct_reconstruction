"""
Project-wide small utilities shared across training, evaluation and plotting.
"""

import json
from pathlib import Path

import torch


def to_4d(t: torch.Tensor, device=None):
    """
    Convert (1, H, W) to (1, 1, H, W) for network input.

    Networks (policy, critic) expect 4D (B, 1, H, W), but LION operators
    and ADMMStep work in 3D (1, H, W). This helper inserts the batch
    dimension when needed, and optionally moves the tensor to a device.

    Args:
        t:      tensor of any rank.
        device: optional device to move the tensor to first.

    Returns:
        The tensor, with a leading batch dimension if it was 3D.
    """
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0) if t.dim() == 3 else t

def project_and_add_noise(gt, op, noise_std, seed=None):
    """
    Forward-project a ground-truth image and add Gaussian sinogram noise.

    LION's tomosipo operator sums mu values along each ray without a path-length
    correction, so the sinogram is not in the same units as the image. SCALE
    maps between them, and the noise is defined on the scaled sinogram:

        SCALE = max(A gt) / max(gt)
        y     = (A gt / SCALE  +  noise_std * std(A gt / SCALE) * eps) * SCALE

    so `noise_std` is a fractional noise level relative to the spread of the
    scaled sinogram (0.05 = 5%).

    Args:
        gt:        ground-truth image, shape (1, H, W).
        op:        LION CT operator.
        noise_std: fractional noise level (e.g. 0.05 = 5%).
        seed:      optional int

    Returns:
        Noisy sinogram, same shape as op.forward(gt).
    """
    # generate clean sinogram
    sino_clean = op.forward(gt)

    # find scaling factor to match gt max
    SCALE = sino_clean.max() / gt.max()

    # scale sinogram
    sino_scaled = sino_clean / SCALE

    # add Gaussian noise to scaled sinogram
    if seed is not None:
        gen = torch.Generator(device=sino_clean.device).manual_seed(int(seed))
        eps = torch.randn(sino_clean.shape, generator=gen,
                          device=sino_clean.device, dtype=sino_clean.dtype)
    else:
        eps = torch.randn_like(sino_clean)

    sino_noisy = sino_scaled + noise_std * sino_scaled.std() * eps

    # rescale back to original units
    return sino_noisy * SCALE

def read_metrics_config(ckpt_dir):
    """
    Read the 'config' dict from a checkpoint directory's metrics_history.json.

    Args:
        ckpt_dir: path to a run's save folder.

    Returns:
        The config dict, or {} if the file is missing or has no config key —
        callers should use .get(key, default) for safe field access.
    """
    # set up path
    ckpt_dir = Path(ckpt_dir)
    metrics_path = ckpt_dir / "metrics_history.json"
    
    # read config
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f).get('config', {})
    

def setup_admm(denoiser_path, device, n_x_steps: int = 6, **admm_kwargs):
    """
    Build the standard ADMM environment used by all eval/plotting scripts.

    Args:
        denoiser_path: path to drunet_gray.pth.
        device:        torch device.
        n_x_steps:     number of gradient descent steps for z-update (default 6).
        **admm_kwargs: passed to ADMMStep.

    Returns:
        (geometry, operator, denoiser, admm_step)
    """
    # make geometry, operator, denoiser, and ADMM step
    from LION.CTtools.ct_utils import make_operator

    from ct_tfpnp.experiments.parallel_beam_ct import experiment
    from ct_tfpnp.models.denoiser import DRUNetDenoiser
    from ct_tfpnp.ct_ops.admm import ADMMStep

    geo = experiment.experiment_params.geometry
    op = make_operator(geo)

    denoiser = DRUNetDenoiser(pretrained_path=denoiser_path).to(device)
    for p in denoiser.parameters():
        p.requires_grad_(False)
    denoiser.eval()

    admm_step = ADMMStep(op=op, denoiser=denoiser, n_x_steps=n_x_steps, **admm_kwargs)
    
    return geo, op, denoiser, admm_step