"""Project-wide small utilities."""

import torch


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