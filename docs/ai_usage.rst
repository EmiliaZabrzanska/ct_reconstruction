AI / LLM usage
============================================================

Large language models (primarily Gemini and Claude) were used to support this project. This page records where, mirroring the declaration in the dissertation.

Areas of use
------------------------------------------------------------

Plotting boilerplate
   Matplotlib scaffolds for the training curves, reconstruction galleries and
   sensitivity sweeps (the ``plot_*`` scripts).

Library usage
   Working out calling conventions for alignment with LION.

Debugging
   Diagnosing the CT-DRUNet calibration mismatch, the ``run_11`` critic explosion, and the
   unit-scale mismatches between LION and TorchRadon in the ADMM z-step.

Docstrings and documentation
   Function and module level docstrings across ``ct_tfpnp``, and autodocumentation using Sphinx.

Test suite
   Editing the ``pytest`` suite to ensure the tests run without a GPU.

Report review
   Redundancy checks, cross-referencing and typo fixes.

Cross-checking against the paper
   Comparing implementation choices with Wei et al. (2022) and consolidating the
   differences into the implementation-differences table.

Verification
------------------------------------------------------------

All model output was reviewed and, where used, edited before being committed to the
repository or incorporated into the report. No model output was accepted without
verification against the source paper, the LION toolbox, or empirical results from
the reproduction itself.

The core experimental design, hyperparameter choices and interpretation of the results are the my own.