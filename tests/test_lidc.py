"""
Tests for ct_tfpnp.datasets.lidc (the parts that do not need LION).
"""

import numpy as np
import pytest
import torch

from ct_tfpnp.datasets.lidc import MU_WATER, from_HU_to_mu, is_lung_slice


class TestIsLungSlice:

    # check if dark regions are detected
    def test_detects_dark_central_regions(self):

        # define image with dark centre
        img = np.ones((64, 64)) * 1.0
        img[16:48, 16:48] = 0.0                  
        assert is_lung_slice(img) is True


