"""
Tests for ct_tfpnp.ct_ops.fbp.
"""

import pytest
import torch

from ct_tfpnp.ct_ops.fbp import calibrate_to_data, fbp, ramp_filter


class TestRampFilter:

    # check shape
    def test_preserves_shape(self, sino):
        assert ramp_filter(sino).shape == sino.shape


class TestFBP:

    # check shape is correct
    def test_output_is_in_the_image_domain(self, sino, op):
        assert fbp(sino, op).shape == op.domain_shape

    # check output is non-negative
    def test_is_non_negative(self, sino, op):
        assert fbp(sino, op).min() >= 0