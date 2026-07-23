# EVT6 Collaboration Handoff

This document records the main files and checks needed to continue EVT6 experiments.

## Core Code

- `main.py`: common training and evaluation entry point.
- `datasets/diting2_evt6.py`: EVT6 dataset implementation.
- `models/SeisMoLLM.py`: pretrained Transformer-based model.
- `tools/report_evt6_results.py`: result reporting utilities.
- `tools/aggregate_evt6_event_level.py`: event-level probability aggregation.

## Reproducibility Notes

- Keep raw datasets outside the repository.
- Record the exact data split, random seed, checkpoint, and command line for each experiment.
- Use `docs/REPRODUCIBILITY.md` as the primary runbook.
- Store model checkpoints in `checkpoints/` locally, but do not commit large binary files.

## Before Release

- Confirm that no raw data, checkpoints, logs, or personal paths are committed.
- Verify that the README, license, citation metadata, and data-access documentation are present.
- Run a syntax check for all Python files before pushing.
