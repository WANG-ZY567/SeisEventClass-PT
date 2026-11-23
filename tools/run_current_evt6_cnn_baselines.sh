#!/usr/bin/env bash
# Re-run the three original waveform CNN baselines on the current EVT6 split.
# Execute on the VS Code server; no source data or existing experiments are changed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DATA:?Set DATA to the prepared EVT6 dataset directory}"
PY="${PY:-python}"
EXP="${EXP:-$CODE/outputs/EVT6_original_CNN_baselines_$(date +%Y%m%d_%H%M%S)}"

command -v "$PY" >/dev/null 2>&1 || test -x "$PY"
test -f "$CODE/main.py"
test -f "$CODE/tools/analyze_evt6_predictions.py"
for split in train val test; do test -f "$DATA/meta_evt6_${split}.csv"; done

"$PY" - "$DATA" <<'PY'
import csv, os, sys
data = sys.argv[1]
expected = {'train': (17502, 1993), 'val': (2075, 248), 'test': (2168, 252)}
seen = set()
for split, (count, event_count) in expected.items():
    path = os.path.join(data, f'meta_evt6_{split}.csv')
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count, (split, len(rows))
    assert {'event_uid', '_npy_path', '_evt6'} <= set(rows[0]), split
    events = {r['event_uid'] for r in rows}
    assert len(events) == event_count, (split, len(events))
    assert not (seen & events), ('event overlap', split)
    seen |= events
    assert {int(r['_evt6']) for r in rows} == set(range(6)), split
    def waveform_path(row):
        path = row['_npy_path'].strip().replace('\\', '/')
        return path if os.path.isabs(path) else os.path.join(data, path)
    missing = [waveform_path(r) for r in rows if not os.path.isfile(waveform_path(r))]
    assert not missing, ('missing waveform', split, len(missing), missing[:3])
    print(f'{split}: {len(rows)} waveforms, {len(events)} events')
PY

mkdir -p "$EXP"
printf 'code=%s\ndata=%s\nseed=20261107\nselection=validation accuracy\n' "$CODE" "$DATA" > "$EXP/protocol.txt"
sha256sum "$DATA"/meta_evt6_{train,val,test}.csv > "$EXP/split_sha256.txt"
cd "$CODE"

for model in CNNsmall_evt6 InceptionTime_evt6 ResNet1D_evt6; do
  echo "===== Training $model ====="
  OUT="$EXP/$model"
  mkdir -p "$OUT"
  "$PY" main.py \
    --mode train_test \
    --model-name "$model" \
    --dataset-name diting2_evt6 \
    --data "$DATA" \
    --log-base "$OUT" \
    --seed 20261107 \
    --device cuda:0 \
    --in-samples 8192 \
    --norm-mode std \
    --augmentation true \
    --batch-size 16 \
    --workers 8 \
    --epochs 200 \
    --patience 30 \
    --best-metric accuracy \
    --optim Adam \
    --weight_decay 0 \
    --loss-name cb_ce \
    --cb-beta 0.995 \
    --cb-samples-per-class 4629,4629,4287,1440,160,2357 \
    --cb-weight-schedule fixed \
    --loss-from-logits false \
    --use-lr-scheduler true \
    --lr-scheduler-name cyclic \
    --lr-scheduler-mode exp_range \
    --base-lr 8e-5 \
    --max-lr 1e-3 \
    --warmup-steps 2000 \
    --down-steps 3000 \
    --save-test-results true \
    --save-test-probs true \
    --use-tensorboard false \
    2>&1 | tee "$OUT/launcher.log"

  mapfile -t predictions < <(find "$OUT" -name test_results_diting2_evt6_test.csv -type f)
  test "${#predictions[@]}" -eq 1
  "$PY" tools/analyze_evt6_predictions.py "${predictions[0]}" \
    --out-dir "$OUT/analysis" 2>&1 | tee "$OUT/analysis.log"
done

echo "Completed: $EXP"
