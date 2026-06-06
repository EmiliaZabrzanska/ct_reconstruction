# Data

Raw and processed data files are stored here but not tracked by git.

## Expected structure
data/
├── raw/            # Original dataset files
├── processed/      # Preprocessed tensors
├── splits/         # Train/val/test split indices
└── debug_subset/   # Small subset for local testing

## Dataset

The TFPnP paper uses the COVID-19 CT lung dataset (Ma et al., 2020) for sparse-view CT experiments. The project brief suggests the Mayo Clinic Low Dose CT Grand Challenge dataset as an alternative, but all experiments in this repository have been done usin the LIDC-IDRI dataset due to it's already strong integration with LION.