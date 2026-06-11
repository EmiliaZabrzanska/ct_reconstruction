"""
ct_tfpnp: Tuning-Free Plug-and-Play for CT Reconstruction

A Python package implementing TFPnP (Wei et al.) adapted for sparse-view CT,
built on top of the LION reconstruction library.

Typical usage:
    from ct_tfpnp.training.solver import TFPnPSolver
    from ct_tfpnp.models.policy import ResNetActor_ADMM
    from ct_tfpnp.models.denoiser import UNetDenoiser2D
"""

from ct_tfpnp.version import __version__
from ct_tfpnp.ct_ops.fbp import fbp, ramp_filter
from ct_tfpnp.ct_ops.admm import ADMMStep

__all__ = ["__version__"]