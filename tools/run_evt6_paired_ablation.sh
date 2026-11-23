#!/usr/bin/env bash
# Matched reference and reviewer-requested ablations, including a fully frozen GPT-2.
set -euo pipefail
export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${DATA:?Set DATA to the prepared EVT6 dataset directory}"
PY="${PY:-python}"
SEED="${SEED:-20261107}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
cd "$ROOT"
[[ -x "$PY" && -f tools/evt6_paired_ablation.py && -d "$DATA" ]] || {
  echo '缺少 Python、实验脚本或数据集，停止。' >&2; exit 1;
}
[[ -f tools/analyze_evt6_predictions.py ]] || { echo '缺少预测分析脚本' >&2; exit 1; }

# The identical split is necessary for pairing with the existing w005 runs.
"$PY" - "$DATA" <<'PY'
from pathlib import Path
import hashlib, sys
root = Path(sys.argv[1])
for split in ('train', 'val', 'test'):
    p = root / ('meta_evt6_' + split + '.csv')
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f'{split} SHA-256: {actual}', flush=True)
PY

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="$OUTPUT_ROOT/EVT6_paired_ablation_seed${SEED}_${STAMP}"
mkdir -- "$OUT"
mkdir -- "$OUT/source_snapshot" "$OUT/audit"
cp -- main.py config.py "$OUT/source_snapshot/"
cp -r -- models training datasets utils "$OUT/source_snapshot/"
cp -- tools/evt6_paired_ablation.py tools/run_evt6_paired_ablation.sh "$OUT/source_snapshot/"
"$PY" - > "$OUT/environment.txt" <<'PY'
import importlib.metadata
import platform
import sys

print('Python:', sys.version.replace('\n', ' '))
print('Executable:', sys.executable)
print('Platform:', platform.platform())
print('\nInstalled distributions:')
items = []
for dist in importlib.metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        items.append((name, dist.version))
for name, version in sorted(items, key=lambda item: item[0].lower()):
    print(f'{name}=={version}')
PY
nvidia-smi > "$OUT/gpu_info.txt" 2>&1 || true
printf '数据=%s\nseed=%s\n变体=full random_lora single_scale pretrained_no_lora pretrained_frozen\n' "$DATA" "$SEED" > "$OUT/experiment_config.txt"
echo "实验目录：$OUT"

for variant in full random_lora single_scale pretrained_no_lora pretrained_frozen; do
  echo "===== 前向、反向及更新检查：$variant ====="
  EVT6_ABLATION="$variant" EVT6_ABLATION_PARAMETER_FILE="$OUT/audit/$variant.json" \
    "$PY" -u tools/evt6_paired_ablation.py --audit 2>&1 | tee "$OUT/audit/$variant.log"
done

# A genuine weight-initialization comparison must have equal parameter counts.
"$PY" - "$OUT/audit" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
r={n: json.loads((p/(n+'.json')).read_text()) for n in
   ('full','random_lora','single_scale','pretrained_no_lora','pretrained_frozen')}
for key in ('total_parameters','trainable_parameters','active_transformer_layers'):
    if r['full'][key] != r['random_lora'][key]:
        raise SystemExit(f'Full vs random parameter mismatch: {key}: '
                         f'{r["full"][key]} != {r["random_lora"][key]}')
if r['full']['trainable_gpt2_shapes'] != r['random_lora']['trainable_gpt2_shapes']:
    raise SystemExit('Full vs random trainable GPT-2 names or shapes differ')
full_non_lora={n.removeprefix('base_model.model.'): shape
               for n,shape in r['full']['trainable_gpt2_shapes'].items()
               if 'lora' not in n.lower()}
if full_non_lora != r['pretrained_no_lora']['trainable_gpt2_shapes']:
    raise SystemExit('LoRA comparison differs in other trainable GPT-2 parameters')
if r['pretrained_frozen']['trainable_gpt2_shapes']:
    raise SystemExit('Frozen GPT-2 contains trainable parameters')
for key in ('n_embd','n_head','n_positions','resid_pdrop','embd_pdrop','attn_pdrop'):
    if r['full']['gpt2_config'].get(key) != r['random_lora']['gpt2_config'].get(key):
        raise SystemExit(f'Full vs random GPT-2 config mismatch: {key}')
for n,d in r.items():
    print(f'{n}: total={d["total_parameters"]}, trainable={d["trainable_parameters"]}, '
          f'layers={d["active_transformer_layers"]}')
print('[OK] 主模型与随机初始化组的参数量一致')
PY

for variant in full random_lora single_scale pretrained_no_lora pretrained_frozen; do
  EXP="$OUT/$variant"
  mkdir -- "$EXP"
  cp -- "$OUT/experiment_config.txt" "$EXP/"
  cp -- "$OUT/audit/$variant.json" "$EXP/model_parameters.json"
  echo
  echo "===== 开始 $variant，seed=$SEED ====="
  export EVT6_ABLATION="$variant"
  unset EVT6_ABLATION_PARAMETER_FILE
  "$PY" -u tools/evt6_paired_ablation.py \
    --mode train_test --model-name SeisMoLLM_evt6_multihead_w005 \
    --dataset-name diting2_evt6 --data "$DATA" --log-base "$EXP" \
    --force-logdir true --device cuda:0 --llm GPT2 --pretrain true --freeze true \
    --LLM_start_layer 0 --LLM_layer_num 3 --d_model 768 --seed "$SEED" \
    --in-samples 8192 --batch-size 16 --workers 8 --epochs 200 --patience 30 \
    --optim Adam --weight-decay 0 --best-metric accuracy \
    --loss-name class_balanced_ce --cb-beta 0.995 \
    --cb-samples-per-class 4629,4629,4287,1440,160,2357 \
    --cb-weight-schedule fixed --use-lr-scheduler true \
    --lr-scheduler-name cyclic --lr-scheduler-mode exp_range --lr-scheduler-gamma 1.0 \
    --base-lr 8e-5 --max-lr 0.001 --warmup-steps 2000 --down-steps 3000 \
    --augmentation true --shift-event-rate 0.2 --add-noise-rate 0.4 \
    --add-gap-rate 0.4 --drop-channel-rate 0.4 --scale-amplitude-rate 0.4 \
    --pre-emphasis-rate 0.4 --generate-noise-rate 0.05 --norm-mode std \
    --save-test-results true --save-test-probs true --tta-times 1 \
    2>&1 | tee "$EXP/launcher.log"
  "$PY" - "$EXP" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
files=sorted(root.glob('*/test_results_diting2_evt6_test.csv'))
if len(files) != 1:
    raise SystemExit(f'Expected exactly one test prediction CSV: {files}')
print(files[0])
(root/'test_prediction_path.txt').write_text(str(files[0])+'\n', encoding='utf-8')
PY
  CSV="$(cat "$EXP/test_prediction_path.txt")"
  "$PY" tools/analyze_evt6_predictions.py "$CSV" 2>&1 | tee "$EXP/analysis.log"
  echo "===== $variant 完成 ====="
done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); result={}
for variant in ('full','random_lora','single_scale','pretrained_no_lora','pretrained_frozen'):
    csv=Path((root/variant/'test_prediction_path.txt').read_text().strip())
    summary=csv.parent/'analysis'/'summary.json'
    d=json.loads(summary.read_text(encoding='utf-8'))
    result[variant]={'trace':d['trace'], 'event_mean_probability':d['event_mean_probability'],
                     'summary':str(summary)}
    t=d['trace']; e=d['event_mean_probability']
    print(f'{variant:20} trace_accuracy={t["accuracy"]:.4f} '
          f'trace_macro_f1={t["macro_f1"]:.4f} '
          f'event_accuracy={e["accuracy"]:.4f} event_macro_f1={e["macro_f1"]:.4f}')
(root/'paired_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('结果与配置：', root)
PY
