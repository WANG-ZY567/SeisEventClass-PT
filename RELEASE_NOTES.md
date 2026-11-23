# Public repository update notes

This release copy aligns the repository with the revised SeisEventClass-PT
manuscript.

## Main changes

- Set `SeisMoLLM_evt6_multihead_w005` as the default paper model.
- Standardized the paper random seed to `20261107`.
- Added exact Linux and Windows training/evaluation entry points.
- Documented the manuscript names of all four matched ablations.
- Updated the class-balanced loss counts to the final training manifest:
  `4629,4629,4287,1440,160,2357`.
- Updated the documented environment to PyTorch 2.7.1.
- Replaced private server paths with environment variables or relative paths.
- Clarified that the manuscript's ComCat result uses strict six-class outputs
  while external reference labels are available only for `eq` and `ep`.
- Removed virtual environments, smoke-test artefacts, caches, raw data,
  checkpoints, and historical experiment outputs from the release copy.
- Added a GitHub Actions workflow and local public-release audit.

## Before publishing

1. Run `python tools/audit_public_release.py`.
2. Confirm the repository URL and author order in `CITATION.cff`.
3. Confirm that dataset-provider requirements are reflected in
   `docs/DATA_ACCESS.md`.
4. Add the final manuscript citation/DOI when available.
5. Tag the release only after the paper configuration has been checked on the
   target server.
