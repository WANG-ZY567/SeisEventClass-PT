#!/usr/bin/env bash
# The published SeisMoLLM architecture, adapted to the six EVT6 classes:
# preserve its convolutional embedder, GPT-2/LoRA stack and classification
# head; change only the head output width from 2 (polarity) to 6 (EVT6).
# Run on the VS Code server. This creates a new experiment and never overwrites
# the existing SeisEventClass-PT result.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DATA:?Set DATA to the prepared EVT6 dataset directory}"
: "${GPT2:?Set GPT2 to the local GPT-2 checkpoint directory}"
PY="${PY:-python}"
EXP="${EXP:-$CODE/outputs/SeisMoLLM_EVT6_singlehead_seed20261107_$(date +%Y%m%d_%H%M%S)_$$}"

command -v "$PY" >/dev/null 2>&1 || test -x "$PY"
test -f "$CODE/main.py"
test -f "$CODE/models/SeisMoLLM.py"
test -f "$CODE/tools/analyze_evt6_predictions.py"
test -f "$GPT2/config.json"
test -f "$GPT2/model.safetensors" || test -f "$GPT2/pytorch_model.bin"
for split in train val test; do test -f "$DATA/meta_evt6_${split}.csv"; done

"$PY" - "$DATA" <<'PY'
import csv
import os
import sys
from collections import Counter

data = sys.argv[1]
expected = {
    'train': (17502, 1993),
    'val': (2075, 248),
    'test': (2168, 252),
}
expected_train_counts = [4629, 4629, 4287, 1440, 160, 2357]
seen_events = set()
for split, (waveforms, events) in expected.items():
    manifest = os.path.join(data, f'meta_evt6_{split}.csv')
    with open(manifest, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == waveforms, (split, 'waveform count', len(rows))
    assert rows and {'event_uid', '_npy_path', '_evt6'} <= rows[0].keys(), split
    event_ids = {row['event_uid'] for row in rows}
    assert len(event_ids) == events, (split, 'event count', len(event_ids))
    assert not (seen_events & event_ids), (split, 'overlapping events')
    seen_events |= event_ids
    counts = Counter(int(row['_evt6']) for row in rows)
    assert set(counts) == set(range(6)), (split, counts)
    if split == 'train':
        assert [counts[i] for i in range(6)] == expected_train_counts, counts
    for row in rows:
        path = row['_npy_path'].strip().replace('\\', '/')
        if not os.path.isabs(path):
            path = os.path.join(data, path)
        assert os.path.isfile(path), ('missing waveform', split, path)
    print(f'[OK] {split}: {len(rows)} waveforms, {len(event_ids)} events, '
          f'counts={[counts[i] for i in range(6)]}')
PY

mkdir "$EXP"
printf 'running\n' > "$EXP/status.txt"
trap 'rc=$?; if ((rc == 0)); then printf "completed\n"; else printf "failed (exit %s)\n" "$rc"; fi > "$EXP/status.txt"' EXIT
export SEISMOLLM_GPT2_PATH="$GPT2"
{
  printf 'experiment=SeisMoLLM-EVT6, adapted from the published architecture\n'
  printf 'source=https://github.com/StarMoonWang/SeisMoLLM\n'
  printf 'architecture=original embedder, GPT-2/LoRA stack, and classification head\n'
  printf 'task_adaptation=change classification output from 2 polarity classes to 6 EVT6 classes\n'
  printf 'comparison=SeisEventClass-PT full model on the same EVT6 data\n'
  printf 'code=%s\ndata=%s\ngpt2=%s\n' "$CODE" "$DATA" "$GPT2"
  printf 'seed=20261107\nselection=validation accuracy\n'
  printf 'test_policy=one evaluation of the validation-selected checkpoint\n'
  printf 'training_protocol=matched to current EVT6 main experiment; CB-CE is an EVT6 training choice\n'
} > "$EXP/protocol.txt"
sha256sum "$DATA"/meta_evt6_{train,val,test}.csv > "$EXP/split_sha256.txt"
sha256sum "$CODE"/main.py "$CODE"/config.py "$CODE"/models/SeisMoLLM.py \
  "$CODE"/tools/analyze_evt6_predictions.py "$GPT2"/config.json \
  > "$EXP/source_sha256.txt"
"$PY" --version > "$EXP/environment.txt" 2>&1
"$PY" -c 'import torch, transformers, peft; print("torch",torch.__version__,"cuda",torch.version.cuda,"gpu",torch.cuda.get_device_name(0)); print("transformers",transformers.__version__,"peft",peft.__version__)' \
  >> "$EXP/environment.txt"

cd "$CODE"
"$PY" main.py \
  --mode train_test \
  --model-name SeisMoLLM_evt6 \
  --dataset-name diting2_evt6 \
  --data "$DATA" \
  --log-base "$EXP" \
  --seed 20261107 \
  --device cuda:0 \
  --llm GPT2 --pretrain true --freeze true \
  --LLM_start_layer 0 --LLM_layer_num 3 --d_model 768 \
  --in-samples 8192 --norm-mode std \
  --augmentation true \
  --shift-event-rate 0.2 --add-noise-rate 0.4 --add-gap-rate 0.4 \
  --drop-channel-rate 0.4 --scale-amplitude-rate 0.4 \
  --pre-emphasis-rate 0.4 --generate-noise-rate 0.05 \
  --batch-size 16 --workers 8 \
  --epochs 200 --patience 30 --best-metric accuracy \
  --optim Adam --weight_decay 0 \
  --loss-name cb_ce --cb-beta 0.995 \
  --cb-samples-per-class 4629,4629,4287,1440,160,2357 \
  --cb-weight-schedule fixed --loss-from-logits false \
  --use-lr-scheduler true --lr-scheduler-name cyclic \
  --lr-scheduler-mode exp_range --lr-scheduler-gamma 1.0 \
  --base-lr 8e-5 --max-lr 1e-3 \
  --warmup-steps 2000 --down-steps 3000 \
  --tta-times 1 --save-test-results true --save-test-probs true \
  --use-tensorboard false \
  2>&1 | tee "$EXP/launcher.log"

mapfile -t predictions < <(find "$EXP" -type f -name test_results_diting2_evt6_test.csv)
test "${#predictions[@]}" -eq 1
"$PY" tools/analyze_evt6_predictions.py "${predictions[0]}" \
  --out-dir "$EXP/analysis" 2>&1 | tee "$EXP/analysis.log"

"$PY" - "$EXP/analysis/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert summary['trace']['num_samples'] == 2168
assert summary['event_mean_probability']['num_samples'] == 252
for level in ('trace', 'event_mean_probability'):
    value = summary[level]
    print(f'{level}: accuracy={value["accuracy"]:.4f}, '
          f'macro_f1={value["macro_f1"]:.4f}, n={value["num_samples"]}')
PY
printf 'Completed: %s\n' "$EXP"
