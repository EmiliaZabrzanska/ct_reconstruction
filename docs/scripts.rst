Scripts
=======

The scripts are thin command-line front-ends, and are not part of the importable package, but are docuented here for completeness.

Setup
-----

.. code-block:: bash

   python scripts/download_denoiser.py

Fetches ``drunet_gray.pth`` (KAIR) into ``results/baselines/``.

Training
--------

.. code-block:: bash

   # TFPnP policy
   python -u scripts/train_tfpnp.py \
       --output_dir results/learned/run_XX \
       --n_train 250 --n_epochs 80

   # FBPConvNet baseline
   python -u scripts/train_fbpconvnet.py \
       --output_dir results/learned/fbpconvnet_pat_250_e80

   # CT-domain DRUNet (optional alternative denoiser prior)
   python -u scripts/train_drunet_ct.py \
       --output_dir results/baselines/drunet_ct

.. important::

   ``--batch_size`` and ``--pi2_batch_size`` must both be greater than 1 due to the policy's batch normalisation layers.

Evaluation and figures
----------------------

``evaluate_run.py`` must be run first as the gallery script relies on the ``per_image_metrics.json`` it writes.

.. code-block:: bash

   python -u scripts/evaluate_run.py --experiment_name run_XX \
       --fbpconvnet_ckpt results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth

   python -u scripts/plot_training_curves.py       --experiment_name run_XX
   python -u scripts/plot_checkpoint_comparison.py --experiment_name run_XX
   python -u scripts/plot_reconstruction_gallery.py --experiment_name run_XX
   python -u scripts/plot_policy_behaviour.py      --experiment_name run_XX
