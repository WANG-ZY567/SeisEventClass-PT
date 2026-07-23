# Open-Source Release Checklist

Use this checklist before uploading the repository to GitHub and before linking it in a manuscript.

## Required by Computers & Geosciences-style software/code guidance

- [x] Public repository is available and documented.
- [x] A clear license is included (`LICENSE`, MIT License).
- [x] English `README.md` provides installation, data preparation, training, evaluation, and citation information.
- [x] Dependencies and computational requirements are documented in `requirements.txt` and `README.md`.
- [x] Raw data, generated arrays, model caches, trained checkpoints, and experiment logs are excluded from the release.
- [x] Data access restrictions and reproduction boundaries are documented in `docs/DATA_ACCESS.md`.
- [x] Minimal train/evaluation commands are provided in `README.md`, `scripts/`, and `docs/REPRODUCIBILITY.md`.
- [x] Output/checkpoint placeholder directories contain README files rather than binary artifacts.
- [x] Repository is distributed as normal source files, not as a single compacted archive.

## Final manual checks before submission

- [ ] Confirm the GitHub URL in the manuscript, cover letter, and README all point to the same public repository.
- [ ] Confirm no private SSH keys, tokens, passwords, or server-specific credentials are included.
- [ ] Confirm no `.h5`, `.hdf5`, `.npy`, `.npz`, `.dat`, large `.csv`, `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `logs/`, `wandb/`, or `gpt2_cache/` files are staged.
- [ ] Run `python main.py --help` in a clean environment.
- [ ] Run `python test_setup.py` after installing dependencies, if the local environment has the required packages.
- [ ] Review remaining non-English comments/messages in code if strict journal compliance is required.

## Suggested repository scope

Keep:

- Core model code: `models/`
- Dataset loaders: `datasets/`
- Training / validation code: `training/`
- Utility code: `utils/`
- Reproducible scripts: `scripts/`, `run_scripts/`
- Data preparation and reporting tools: `tools/`
- Method, data, and reproducibility documents: `docs/`
- README figures: `figures/`

Exclude:

- `logs/`, historical experiment folders, and `model_backup.py`
- `gpt2_cache/`
- `__pycache__/`
- Generated shell-argument files such as `--batch_size`, `--lr`, `--epochs`
- Large data, checkpoints, temporary outputs, and archive files
