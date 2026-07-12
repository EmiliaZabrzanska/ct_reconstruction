"""
LIDC-IDRI dataset for TFPnP CT training.

The training pipeline loads images via get_lion_split, which delegates to
LION's built-in LIDC_IDRI dataloader and returns ground-truth tensors in
LION's native µ scaling (water normalised to µ = 1.0).
"""

import torch
import numpy as np
from pathlib import Path

# LION-native attenuation scaling
MU_WATER = 1.0 



def from_HU_to_mu(img_hu, mu_water=MU_WATER):
    """
    Convert Hounsfield Units to LION-native linear attenuation mu, clipped at 0.

    Args:
        img_hu:   array in HU.
        mu_water: attenuation of water in the target scaling.

    Returns:
        Array in mu units, non-negative.
    """
    return np.clip((img_hu / 1000.0 + 1.0) * mu_water, 0, None)


def is_lung_slice(img, dark_threshold=None, min_dark_fraction=0.15):
    """
    Check if a CT slice contains lungs (dark air-filled regions in centre).

    Args:
        img:                 2D array or tensor (H, W) in µ or HU units.
        dark_threshold:      values below this are considered air.
        min_dark_fraction:   minimum fraction of centre that must be dark.

    Returns:
        True if the slice likely contains lungs.
    """
    # convert to numpy array
    if isinstance(img, torch.Tensor):
        img = img.squeeze().cpu().numpy()
    img = np.asarray(img).squeeze()

    # determine dark threshold if not provided
    if dark_threshold is None:
        dark_threshold = img.max() * 0.2

    # set image shape and centre
    h, w = img.shape
    centre = img[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5]

    # check if dark fraction exceeds minimum threshold
    return float((centre < dark_threshold).mean()) > min_dark_fraction


def get_lion_split(split="train", geometry=None, device="cuda"):
    """
    Load an LIDC-IDRI split via LION's built-in dataloader.

    Returns ground-truth images in LION's native µ scaling.

    Args:
        split:      "train", "validation" or "test".
        geometry:   LION Geometry (LION needs it to construct the dataset).
        device:     device to place the images on.

    Returns:
        List of ground-truth tensors, each of shape (1, H, W), in mu units.
    """
    from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

    # load the LION dataset
    lion_dataset = LIDC_IDRI(mode=split, geometry_parameters=geometry)
    n = len(lion_dataset)

    images = []
    for i in range(n):

        # get ground truth images
        _, gt = lion_dataset[i]
        if gt.dim() == 2:

            # add channel dimension
            gt = gt.unsqueeze(0)
        images.append(gt.float().to(device))

    print(f"LION {split} split: {len(images)} images, "
          f"range [{images[0].min():.4f}, {images[0].max():.4f}]")
    
    return images