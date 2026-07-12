ct_tfpnp
========

A reproduction of Tuning-free Plug-and-Play (TFPnP) proximal algorithms (Wei et al., *JMLR* 23, 2022) for 30-view parallel-beam sparse-view CT reconstruction, 
built on the `LION <https://github.com/CambridgeCIA/LION>`_ toolbox and evaluated on LIDC-IDRI.

A reinforcement-learning policy chooses the denoising strength :math:`\sigma`, the ADMM penalty :math:`\mu`, and the stopping time, so that plug-and-play ADMM does not require hand tuning.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/ct_ops
   api/models
   api/training
   api/optimisers
   api/evaluation
   api/datasets
   api/experiments
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Usage

   scripts

.. toctree::
   :maxdepth: 1
   :caption: About

   ai_usage

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
