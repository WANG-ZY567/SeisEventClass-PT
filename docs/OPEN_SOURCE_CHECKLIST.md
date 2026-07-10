# Open-Source Release Checklist

Use this checklist before uploading the repository to GitHub.

## Must Check

- [ ] No private SSH keys, tokens, passwords, or server paths are included.
- [ ] No raw datasets, generated `.h5`, `.npy`, `.npz`, `.dat`, or large CSV files are included.
- [ ] No checkpoints, GPT-2 local cache, TensorBoard logs, or `wandb` runs are included.
- [ ] `python main.py --help` works in a clean environment.
- [ ] `pip install -r requirements.txt` is sufficient, or missing system dependencies are documented.
- [ ] README gives a minimal training and evaluation command.
- [ ] Dataset access and preprocessing are documented.
- [ ] License and citation are correct.

## Suggested Repository Scope

Keep:

- Core model code: `models/`
- Dataset loaders: `datasets/`
- Training / validation code: `training/`
- Utility code: `utils/`
- Reproducible scripts: `run_scripts/`, selected root `*.sh`
- Method and dataset documents: `docs/`
- Figures used by README: `figures/`

Exclude:

- `logs/`, nested historical experiment folders, and `model_backup.py`
- `gpt2_cache/`
- `__pycache__/`
- Generated root files such as `--batch_size`, `--lr`, `--epochs`
- Large data, checkpoints, and temporary outputs
