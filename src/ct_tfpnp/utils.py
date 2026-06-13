"""Project-wide small utilities."""

import json
import torch
from pathlib import Path


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