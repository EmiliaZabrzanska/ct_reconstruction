"""
30-view parallel-beam sparse CT experiment (LION Experiment subclass).
"""

import numpy as np
import math
from LION.experiments.ct_experiments import Experiment
from LION.CTtools.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter


class ParallelBeamSparseViewCT(Experiment):
    """
    30-view parallel beam CT experiment following LION's geometry convention.

        image_size    = image_shape     (unit spacing — 1 unit per pixel)
        detector_size = detector_shape  (1 unit per detector)
    
    which gives full 512x512 image coverage with no truncation and keeps the
    operator scaling consistent between forward, adjoint and FBP.
    """

    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):

        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        """
        Build the geometry, noise and data-loader parameters for this experiment.
        """
        # set up geometry parameters
        param = LIONParameter()
        param.name = "TFPnP 30-view parallel beam sparse CT"

        # define shape and detector count
        image_shape = [1, 512, 512]
        n_detectors = math.ceil(image_shape[1] * math.sqrt(2))  # 725

        param.geometry = Geometry(
            image_shape=image_shape,
            image_size=image_shape,              # unit spacing n
            detector_shape=[1, n_detectors],     
            detector_size=[1, n_detectors],      # unit detector spacing
            dso=image_shape[1] * 2,              # 1024 — unused in parallel mode
            dsd=image_shape[1] * 4,              # 2048 — unused in parallel mode
            mode="parallel",
            angles=np.linspace(0, np.pi, 30, endpoint=False),
        )

        # physical noise model (currently unused but implemented for consistency with LION)
        param.noise_params            = LIONParameter()
        param.noise_params.I0         = 1000
        param.noise_params.sigma      = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(dataset, geometry=param.geometry)
        
        return param
    
# set convenient alias from ct_tfpnp.experiments.parallel_beam_ct import experiment
experiment = ParallelBeamSparseViewCT()