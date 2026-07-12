# Low-Dose CT Reconstruction

Reproduction of TFPnP (Tuning-free Plug-and-Play; Wei et al., 2022) for
sparse-view CT reconstruction, carried out as part of the MPhil in Data Intensive
Science at the University of Cambridge.

## Project overview

- **Problem.** CT reconstruction from sparse-view measurements is ill-posed, giving
  noise and streak artefacts.
- **Approach.** Reproduce TFPnP, which uses reinforcement learning to tune the
  parameters (denoising strength $\sigma$, penalty $\mu$, stopping time) of plug-and-play ADMM,
  per image and per iteration.
- **Setting.** 30-view parallel-beam geometry, LIDC-IDRI, built on the
  [LION toolbox](https://github.com/CambridgeCIA/LION).

## Installation

```bash
conda env create -f environment.yml
conda activate mphil_ct
pip install -e .
```

`environment.yml` targets the CSD3 compute environment (linux-64, CUDA 12.1). It
installs LION, tomosipo and astra, which are needed to run the pipeline. The test
suite skips the LION-dependent tests, so a lighter local env still runs tests and builds the docs.

Then fetch the pretrained denoiser:

```bash
python scripts/download_denoiser.py
```

## Quick start

```bash
# Train the TFPnP policy
python scripts/train_tfpnp.py --output_dir results/learned/run_XX --n_train 250 --n_epochs 80

# Evaluate + make figures (evaluate_run must run first — the gallery reads its JSON)
python scripts/evaluate_run.py --experiment_name run_XX
python scripts/plot_training_curves.py       --experiment_name run_XX
python scripts/plot_checkpoint_comparison.py --experiment_name run_XX
python scripts/plot_reconstruction_gallery.py --experiment_name run_XX
python scripts/plot_policy_behaviour.py      --experiment_name run_XX
```

On CSD3 these are driven by the batch scripts in `slurm/` (see `slurm/README.md`).

## Tests and docs

```bash
pytest                     
cd docs && make html    
```

## Repository structure

```
eaz21/
├── pyproject.toml
├── environment.yml
├── src/ct_tfpnp/          # source package
│   ├── ct_ops/            # FBP, TV, ADMM step on LION's operator
│   ├── datasets/          # LIDC loading and HU conversion
│   ├── models/            # policy, critic, DRUNet denoiser, wrappers
│   ├── training/          # environment, RL losses, replay buffer
│   ├── optimisers/        # TFPnPSolver
│   ├── evaluation/        # PSNR / SSIM / HaarPSI metrics
│   └── experiments/       # parallel-beam geometry 
├── scripts/               # CLI entry points       
├── slurm/                 # CSD3 batch jobs         
├── tests/                 # pytest suite            
├── notebooks/             # exploration and initial development (now outdated)    
├── docs/                  # Sphinx site and paper/brief PDFs
├── data/                  # LIDC subset (not tracked)
├── results/metrics/       # per-run metrics CSV/JSON (tracked)
├── figures/               # report figures
└── report/                # report and executive summary PDFs
```

## AI / LLM usage

Large language models (primarily Gemini and Claude) were used to support this project, in these areas:

- **Plotting boilerplate:** Matplotlib styling for the training curves, reconstruction galleries and sensitivity sweeps.
- **Library usage:** working out calling conventions for aligning functions with LION.
- **Debugging:** diagnosing the CT-DRUNet calibration mismatch, the `run_11` critic explosion, and the LION/TorchRadon scale mismatch in the ADMM z-step.
- **Docstrings and documentation:** creation of function and module level docstrings across `ct_tfpnp`, and the Sphinx site.
- **Test suite:** initial generation of the `pytest` suite.
- **Report review:** performing redundancy checks, cross-referencing, typo fixes, and helping with correct LaTex code for mathematical expressions.
- **Cross-checking against the paper:** comparing implementation choices with Wei et al. (2022) to ensure no design choices were missed.

All model output was reviewed and, where used, edited before being committed or incorporated into the report. No model output was accepted without verifying
against the source paper, the LION toolbox, or results from the reproduction itself. The main experimental design, hyperparameter choices and interpretation of the results are my own.

## Licence

MIT (see `LICENSE`).

## Acknowledgements

Supervised by **Dr Ander Biguri** (DAMTP, University of Cambridge). 
Built on the [LION toolbox](https://github.com/CambridgeCIA/LION).