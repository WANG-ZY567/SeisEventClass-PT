# DiTing 2.0 Usage Notes

This note describes how this repository expects DiTing 2.0 waveform data to be organized.

## Data Access

DiTing 2.0 is an external seismic waveform dataset and is not redistributed in this repository. Please obtain the dataset from the official provider or from the access channel cited in the manuscript.

## Expected Inputs

The training and evaluation scripts expect preprocessed waveform arrays and metadata tables prepared from DiTing 2.0. Typical local inputs include:

- fixed-length three-component waveform arrays;
- event labels mapped to the EVT6 label space;
- metadata files containing event identifiers, station identifiers, split names, and optional phase picks.

Large data files should stay outside the Git repository. Local paths can be passed through command-line arguments or configured in scripts.

## Recommended Workflow

1. Download DiTing 2.0 from the official source.
2. Prepare event metadata and waveform files according to the target task.
3. Use the scripts in `tools/` to create EVT3, EVT5, EVT6, picking-conditioned, or transfer-test subsets when needed.
4. Launch training or evaluation with the scripts in `scripts/` or `run_scripts/`.

For general data-availability information, see `docs/DATA_ACCESS.md`.
