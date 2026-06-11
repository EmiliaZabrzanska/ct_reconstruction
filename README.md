# Low-Dose CT Reconstruction

Reproduction of TFPnP (Tuning-free Plug-and-Play) for sparse-view CT reconstruction,
carried out as part of the MPhil in Data Intensive Science at the University of Cambridge.

## Project overview

- **Problem**: CT reconstruction from sparse-view measurements is ill-posed, leading to noise and artefacts.
- **Approach**: Reproduce the TFPnP method (Wei et al., JMLR 2022) which uses reinforcement learning to automatically tune PnP algorithm parameters.
- **Goal**: Match the paper's sparse-view CT results and integrate the method into the LION toolbox.

## Installation

```bashconda env create -f environment.yml
conda activate ct_recon
pip install -e .

## Repository structurect_reconstruction/
├── pyproject.toml
├── environment.yml
├── docs/                  # Paper PDF, project brief
├── notebooks/             # Exploration and results notebooks
├── report/                # LaTeX report
├── ct_tfpnp/              # Source package
│   ├── ct_ops/            # Forward/adjoint CT operators
│   ├── datasets/          # Data loading and preprocessing
│   ├── models/            # TFPnP policy and denoiser
│   ├── training/          # Training loops
│   └── evaluation/        # Metrics and visualisation
├── scripts/               # CLI entry points
├── tests/                 # Test suite
├── data/                  # Raw/processed data (not tracked)
├── results/               # Outputs and checkpoints (not tracked)
└── figures/               # Figures for report

## Licence

MIT

## Acknowledgements

Supervised by **Dr Ander Biguri** (DAMTP, University of Cambridge).
Built on the [LION toolbox](https://github.com/CambridgeCIA/LION).