"""
Tests for ct_tfpnp.evaluation.metrics.
"""

import math

import pytest
import torch

from ct_tfpnp.evaluation import metrics as M


class TestDataRange:

    # check data range default
    def test_default_is_peak_to_peak(self, gt):
        assert M.default_data_range(gt) == pytest.approx(float(gt.max() - gt.min()))


class TestPSNR:

    # check noise degradation
    def test_more_noise_means_lower_psnr(self, gt):

        # define low and high noise
        low = (gt + 0.01 * torch.randn_like(gt))
        high = (gt + 0.20 * torch.randn_like(gt))

        # check psnr
        assert M.psnr_np(gt, low) > M.psnr_np(gt, high)

    # check differentiability 
    def test_is_differentiable(self, gt):

        # define recon and target
        recon = M._as_4d(gt.clone()).requires_grad_(True)
        target = M._as_4d(gt + 0.1)

        # find gradient of osnr
        M.psnr(recon, target).backward()

        assert recon.grad is not None and recon.grad.abs().sum() > 0


class TestSSIM:

    # check noise degradation
    def test_more_noise_means_lower_ssim(self, gt):
        low = (gt + 0.01 * torch.randn_like(gt)).clamp(min=0)
        high = (gt + 0.20 * torch.randn_like(gt)).clamp(min=0)
        assert M.ssim_np(gt, low) > M.ssim_np(gt, high)



class TestLsScale:

    # check least-squares scale factor is correct
    def test_is_the_least_squares_optimum(self, gt):

        # define recon
        recon = (gt * 0.4 + 0.1 * torch.randn_like(gt))

        # define least-squares optimum
        best = M.ls_scale(gt, recon)
        best_err = ((gt - best) ** 2).sum()
        
        # Any other scalar must do worse
        for alpha in (0.5, 0.9, 1.1, 2.0):
            other = best * alpha
            assert ((gt - other) ** 2).sum() >= best_err - 1e-6
