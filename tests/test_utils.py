"""
Tests for ct_tfpnp.utils.
"""

import json

import pytest
import torch

from ct_tfpnp.utils import project_and_add_noise, read_metrics_config, to_4d


class TestTo4d:

    # test if 3D input is converted to 4D by adding a batch dimension
    def test_adds_batch_dim_to_3d(self):
        assert to_4d(torch.zeros(1, 8, 8)).shape == (1, 1, 8, 8)


class TestProjectAndAddNoise:

    # test that shape is preserved
    def test_shape_matches_forward(self, gt, op):

        # forward project
        y = project_and_add_noise(gt, op, 0.05)

        # check shape matches
        assert y.shape == op.forward(gt).shape

    # test noise is added correctly
    def test_noise_level_is_fractional_wrt_sinogram_std(self, gt, op):

        # create sinogram and y
        sino = op.forward(gt)
        y = project_and_add_noise(gt, op, 0.05, seed=0)

        # find std of residual
        residual_std = (y - sino).std().item()

        # check residual std matches nosie
        assert residual_std == pytest.approx(0.05 * sino.std().item(), rel=0.25)

    # test zero noise 
    def test_zero_noise_returns_clean_sinogram(self, gt, op):
        assert torch.allclose(project_and_add_noise(gt, op, 0.0), op.forward(gt))


class TestReadMetricsConfig:

    # check that config is read correctly
    def test_reads_config(self, tmp_path):

        # write fake config
        cfg = {"m": 5, "N": 6, "sigma_range": [1.0, 5.0]}
        (tmp_path / "metrics_history.json").write_text(json.dumps({"config": cfg}))

        # check config is read correctly
        assert read_metrics_config(tmp_path) == cfg