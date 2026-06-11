"""
LIDC-IDRI dataset for TFPnP CT training.

Provides two interfaces:
1. LIDCSinogramDataset — wraps LION's LIDC_IDRI for DataLoader use
2. load_training_images() — simple list of tensors for notebook use
"""

import torch
import numpy as np
import math
from pathlib import Path
from torch.utils.data import Dataset


# ── Constants ─────────────────────────────────────────────────────────────

MU_WATER = 0.2  # linear attenuation of water at ~70 keV (cm⁻¹)


# ── HU conversion ────────────────────────────────────────────────────────

def from_HU_to_mu(img_hu, mu_water=MU_WATER):
    """Convert Hounsfield Units to linear attenuation (µ cm⁻¹)."""
    return np.clip((img_hu / 1000.0 + 1.0) * mu_water, 0, None)


# ── Lung slice detection ─────────────────────────────────────────────────

def is_lung_slice(img, dark_threshold=None, min_dark_fraction=0.15):
    """
    Check if a CT slice contains lungs (dark air-filled regions in centre).

    Args:
        img: 2D array or tensor (H, W) in µ or HU units
        dark_threshold: values below this are considered air.
                        Auto-set to 20% of image max if None.
        min_dark_fraction: minimum fraction of centre that must be dark

    Returns:
        True if the slice likely contains lungs
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()
    img = np.asarray(img).squeeze()

    if dark_threshold is None:
        dark_threshold = img.max() * 0.2

    h, w = img.shape
    centre = img[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5]
    return float((centre < dark_threshold).mean()) > min_dark_fraction


# ── Simple image loader (for notebooks) ──────────────────────────────────

def load_training_images(data_dir, n_patients=None, device="cuda",
                          lung_only=False):
    """
    Load LIDC-IDRI slices as a list of (1, H, W) tensors in µ units.

    Args:
        data_dir:    path to processed LIDC-IDRI directory
        n_patients:  max patients to load (None = all)
        device:      torch device
        lung_only:   if True, only return slices containing lungs

    Returns:
        list of tensors, each shape (1, 512, 512)
    """
    data_dir = Path(data_dir)
    patients = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if n_patients is not None:
        patients = patients[:n_patients]

    images = []
    for pat_dir in patients:
        slices = sorted(pat_dir.glob("*.npy"))
        if not slices:
            continue
        # Pick middle slice (most likely to contain lung)
        s = slices[len(slices) // 2]
        img_hu = np.load(s).astype(np.float32)
        img_mu = from_HU_to_mu(img_hu)

        if lung_only and not is_lung_slice(img_mu):
            continue

        images.append(torch.tensor(img_mu, device=device).unsqueeze(0))

    return images


# helper to load LION's built-in train/val/test splits of LIDC-IDRI for TFPnP training

def get_lion_split(split="train", geometry=None, device="cuda"):
    """
    Load LIDC-IDRI images using LION's built-in train/test/validation split.
    Returns images in LION's native µ units (water ≈ 1.0).
    """
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
    
    lion_dataset = LIDC_IDRI(
        mode=split,
        geometry_parameters=geometry,
    )
    
    images = []
    indices = []
    for i in range(len(lion_dataset)):
        _, gt = lion_dataset[i]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        gt = gt.float().to(device)
        images.append(gt)
        indices.append(i)
    
    print(f"LION {split} split: {len(images)} images, "
          f"range [{images[0].min():.4f}, {images[0].max():.4f}]")
    return images, indices


# Note: LIDCSinogramDataset uses Poisson noise and is not used in the
# TFPnP pipeline, which generates Gaussian noise on-the-fly in
# collect_episode(). Kept for potential future use with other methods.