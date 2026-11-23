#!/usr/bin/env bash
# Run six-class published-code baselines on the frozen S-eligible split.
set -euo pipefail

: "${ROOT:?Set ROOT to the external baseline workspace}"
: "${DATA:?Set DATA to the prepared S-eligible EVT6 dataset directory}"
LOW_VRAM=${LOW_VRAM:-0}
if [[ "$LOW_VRAM" == 1 ]]; then
  EXP="$ROOT/experiments/evt6_svalid_baselines_acc200_pat30_lowvram"
  CAP_BATCH=16
  CAP_ACCUM=4
  MIN_FREE_MIB=4096
else
  EXP="$ROOT/experiments/evt6_svalid_baselines_acc200_pat30"
  CAP_BATCH=64
  CAP_ACCUM=1
  MIN_FREE_MIB=8192
fi
CAPS="$ROOT/DiTing2.0-seismic-classification-main"
CAP_FIXED="$ROOT/data/diting2_evt6_capsnet_mfcc_svalid_fixedp_v1"
CAP_PS="$ROOT/data/diting2_evt6_capsnet_mfcc_svalid_paperps_v1"

source "$ROOT/.venvs/seisevent/bin/activate"
mkdir -p "$EXP"
echo "Training mode: LOW_VRAM=$LOW_VRAM, CapsNet batch=$CAP_BATCH x $CAP_ACCUM"

cd "$CAPS"
python evt6_capsnet_baseline/prepare_evt6_mfcc.py \
  --data-dir "$DATA" --cache-dir "$CAP_FIXED" \
  --window-policy fixed_p --workers 8
python evt6_capsnet_baseline/prepare_evt6_mfcc.py \
  --data-dir "$DATA" --cache-dir "$CAP_PS" \
  --window-policy paper_ps --workers 8

export DATA CAP_FIXED CAP_PS
python - <<'PY'
import json
import os
from pathlib import Path

expected = {'train': 16604, 'val': 1852, 'test': 1881}
reports = {
    'capsnet_fixedp': (Path(os.environ['CAP_FIXED']) / 'mfcc_preparation_summary.json', 'fixed_p'),
    'capsnet_paperps': (Path(os.environ['CAP_PS']) / 'mfcc_preparation_summary.json', 'p_to_s'),
}
for name, (path, policy) in reports.items():
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['source_data_dir'] == os.environ['DATA'], (name, 'wrong source data')
    assert all(value == 0 for value in report['event_overlap'].values()), (name, 'event overlap')
    counts = {split: int(report['splits'][split]['records']) for split in expected}
    assert counts == expected, (name, counts)
    if policy:
        for split, count in expected.items():
            assert report['splits'][split]['window_policies'] == {policy: count}, (name, split)
    print(f'{name}: records={counts}, overlap={report["event_overlap"]}', flush=True)
PY

check_gpu_space() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU free memory is ${free_mib} MiB; need at least ${MIN_FREE_MIB} MiB before training." >&2
    echo "Caches are ready. Re-run this script when GPU memory is available." >&2
    exit 3
  fi
}

has_completed_run() {
  local output_dir=$1
  [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 2 -maxdepth 2 -name run_summary.json -print -quit)" ]]
}

cd "$CAPS"
if has_completed_run "$EXP/capsnet_fixedp"; then
  echo 'capsnet_fixedp already complete; skipping.'
else
  check_gpu_space
  python evt6_capsnet_baseline/train_evt6_capsnet.py \
    --cache-dir "$CAP_FIXED" --output-dir "$EXP/capsnet_fixedp" \
    --seed 20261105 --batch-size "$CAP_BATCH" --accumulation-steps "$CAP_ACCUM" --workers 8 \
    --epochs 200 --patience 30 --balance none \
    --best-metric accuracy --device cuda
fi

if has_completed_run "$EXP/capsnet_paperps"; then
  echo 'capsnet_paperps already complete; skipping.'
else
  check_gpu_space
  python evt6_capsnet_baseline/train_evt6_capsnet.py \
    --cache-dir "$CAP_PS" --output-dir "$EXP/capsnet_paperps" \
    --seed 20261105 --batch-size "$CAP_BATCH" --accumulation-steps "$CAP_ACCUM" --workers 8 \
    --epochs 200 --patience 30 --balance none \
    --best-metric accuracy --device cuda
fi

echo 'BASELINES_FINISHED'
