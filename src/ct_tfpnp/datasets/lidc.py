"""
LIDC-IDRI dataset for TFPnP CT training.

The training pipeline loads images via get_lion_split, which delegates to
LION's built-in LIDC_IDRI dataloader and returns ground-truth tensors in
LION's native µ scaling (water normalised to µ = 1.0).
"""

import torch
import numpy as np
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────

MU_WATER = 1.0  # LION-native scaling (physical value at 70 keV is ~0.2 cm⁻¹)


# ── HU conversion ────────────────────────────────────────────────────────

def from_HU_to_mu(img_hu, mu_water=MU_WATER):
    """Convert Hounsfield Units to LION-native linear attenuation µ, clipped at 0."""
    return np.clip((img_hu / 1000.0 + 1.0) * mu_water, 0, None)


# ── Lung slice detection ─────────────────────────────────────────────────

def is_lung_slice(img, dark_threshold=None, min_dark_fraction=0.15):
    """
    Check if a CT slice contains lungs (dark air-filled regions in centre).

    Args:
        img: 2D array or tensor (H, W) in µ or HU units
        dark_threshold: values below this are considered air.
                        Auto-set to 20% of image max if None.
        min_dark_fraction: minimum fraction of centre that must be dark.

    Returns:
        True if the slice likely contains lungs.
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()
    img = np.asarray(img).squeeze()

    if dark_threshold is None:
        dark_threshold = img.max() * 0.2

    h, w = img.shape
    centre = img[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5]
    return float((centre < dark_threshold).mean()) > min_dark_fraction


# ── LION split loader (canonical training-data interface) ────────────────

def get_lion_split(split="train", geometry=None, device="cuda"):
    """
    Load an LIDC-IDRI split via LION's built-in dataloader.

    Returns ground-truth images in LION's native µ scaling (water ≈ 1.0).
    """
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

    lion_dataset = LIDC_IDRI(mode=split, geometry_parameters=geometry)

    images, indices = [], []
    for i in range(len(lion_dataset)):
        _, gt = lion_dataset[i]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        images.append(gt.float().to(device))
        indices.append(i)

    print(f"LION {split} split: {len(images)} images, "
          f"range [{images[0].min():.4f}, {images[0].max():.4f}]")
    return images, indices