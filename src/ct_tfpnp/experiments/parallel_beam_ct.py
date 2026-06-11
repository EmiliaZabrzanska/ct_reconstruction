import numpy as np
import math
from LION.experiments.ct_experiments import Experiment
from LION.CTtools.ct_geometry import Geometry
from LION.utils.parameter import LIONParameter
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI


class ParallelBeamSparseViewCT(Experiment):
    """
    30-view parallel beam CT experiment following LION's geometry convention.

    Follows LION's parallel_sparse_view_parameters pattern:
        image_size = image_shape   (unit spacing — 1 unit per pixel)
        detector_size = detector_shape  (1 unit per detector)

    This ensures:
    - Full 512×512 image coverage (no truncation)
    - FBP normalisation works correctly with lion_fbp
    - Consistent operator scaling throughout

    Deviation from Wei et al. 2022: we use 512 detectors rather than 182.
    The sparse reconstruction challenge comes primarily from 30 angles (vs
    the typical 360+), not detector count. Using 512 detectors ensures the
    spatial FOV is fully covered, isolating the angular sparsity effect.
    """

    def __init__(self, experiment_params=None, dataset="LIDC-IDRI", datafolder=None):
        super().__init__(experiment_params, dataset, datafolder)

    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "TFPnP 30-view parallel beam sparse CT"

        image_shape = [1, 512, 512]
        n_detectors = math.ceil(image_shape[1] * math.sqrt(2))  # 724

        param.geometry = Geometry(
            image_shape=image_shape,
            image_size=image_shape,              # unit spacing — matches LION convention
            detector_shape=[1, n_detectors],     
            detector_size=[1, n_detectors],     
            dso=image_shape[1] * 2,              # 1024
            dsd=image_shape[1] * 4,              # 2048
            mode="parallel",
            angles=np.linspace(0, np.pi, 30, endpoint=False),
        )

        param.noise_params = LIONParameter()
        param.noise_params.I0    = 1000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        param.data_loader_params = Experiment.get_dataset_parameters(
            dataset, geometry=param.geometry
        )
        return param
    
# Convenience instance — importable as:
#   from ct_tfpnp.experiments.parallel_beam_ct import experiment
experiment = ParallelBeamSparseViewCT()