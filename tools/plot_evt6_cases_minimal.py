#!/usr/bin/env python3
"""Create six reproducibly selected waveforms for the current EVT6 test set."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data, prediction_file, out = map(Path, sys.argv[1:4])
seed = 20261107
names = ('eq', 'ep', 'ss', 'sp', 'se', 'ot')
pairs = ((3, 3), (3, 0), (4, 4), (4, 0), (5, 5), (5, 2))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rank(purpose, value):
    return hashlib.sha256(f'{seed}|{purpose}|{value}'.encode()).hexdigest()

manifest = data / 'meta_evt6_test.csv'
meta = pd.read_csv(manifest, dtype={'trace_uid': str, 'event_uid': str}, low_memory=False)
pred = pd.read_csv(prediction_file, dtype={'trace_uid': str}, low_memory=False)
required_meta = {'trace_uid', 'event_uid', '_npy_path', '_evt6', 'p_pick', 's_pick'}
prob_cols = [f'prob_evt6_{i}' for i in range(6)]
required_pred = {'trace_uid', 'tgt_evt6', 'pred_evt6', *prob_cols}
if required_meta - set(meta) or required_pred - set(pred):
    raise ValueError(f'Missing columns: meta={required_meta-set(meta)}, prediction={required_pred-set(pred)}')
if len(meta) != 1948 or len(pred) != 1948:
    raise ValueError(f'Expected 1948 test records, got {len(meta)} and {len(pred)}')
if meta.trace_uid.isna().any() or pred.trace_uid.isna().any() or meta.trace_uid.duplicated().any() or pred.trace_uid.duplicated().any():
    raise ValueError('Missing or duplicate trace_uid')
if meta.event_uid.isna().any() or set(meta.trace_uid) != set(pred.trace_uid):
    raise ValueError('Missing event_uid or test identifiers do not match')
df = meta.merge(pred[['trace_uid', 'tgt_evt6', 'pred_evt6'] + prob_cols], on='trace_uid', validate='one_to_one')
for col in ('_evt6', 'tgt_evt6', 'pred_evt6'):
    numeric = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all() or not numeric.between(0, 5).all():
        raise ValueError(f'Invalid six-class labels in {col}')
    df[col] = numeric.astype(int)
if not (df['_evt6'] == df['tgt_evt6']).all():
    raise ValueError('Test and prediction labels differ')
prob = df[prob_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
if not np.isfinite(prob).all() or np.any(prob < -1e-6) or np.any(prob > 1 + 1e-6) or not np.allclose(prob.sum(axis=1), 1, atol=1e-3):
    raise ValueError('Invalid saved probabilities')
if not (prob.argmax(axis=1) == df['pred_evt6'].to_numpy()).all():
    raise ValueError('Predicted labels do not match probability argmax')

groups = []
for target, predicted in pairs:
    group = df[(df.tgt_evt6 == target) & (df.pred_evt6 == predicted)]
    events = sorted(group.event_uid.unique(), key=lambda e: rank(f'{target}-{predicted}-event', e))
    if not events:
        raise ValueError(f'No test cases for {names[target]} -> {names[predicted]}')
    print(f'{names[target]} -> {names[predicted]}: {len(group)} waveforms, {len(events)} events', flush=True)
    groups.append((target, predicted, group, events))

def choose(i, used):
    if i == len(groups):
        return []
    for event in groups[i][3]:
        if event not in used:
            rest = choose(i + 1, used | {event})
            if rest is not None:
                return [event] + rest
    return None

events = choose(0, set())
if events is None:
    raise ValueError('Cannot select all six cases from distinct events')
selected = []
for (target, predicted, group, _), event in zip(groups, events):
    subset = group[group.event_uid == event].set_index('trace_uid')
    uid = sorted(subset.index, key=lambda x: rank(f'{target}-{predicted}-trace', x))[0]
    item = subset.loc[uid].to_dict()
    item['trace_uid'] = uid
    selected.append(item)

plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(13, 10.6))
outer = fig.add_gridspec(3, 2, left=.075, right=.985, bottom=.075, top=.925, wspace=.24, hspace=.48)
records = []
for i, item in enumerate(selected):
    path = Path(str(item['_npy_path']).strip())
    if not path.is_absolute():
        path = data / path
    path = path.resolve(strict=True)
    wave = np.load(path, allow_pickle=False).astype(np.float32, copy=True)
    if wave.shape != (3, 8192) or not np.isfinite(wave).all():
        raise ValueError(f'Invalid waveform: {path}: {wave.shape}')
    p = float(item['p_pick'])
    s = float(item['s_pick']) if pd.notna(item['s_pick']) else float('nan')
    if not np.isfinite(p) or not 0 <= p < 8192:
        raise ValueError(f'Invalid P pick: {item["trace_uid"]}')
    wave -= wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    std[std == 0] = 1
    wave /= std
    if not np.isfinite(wave).all():
        raise ValueError(f'Non-finite normalized waveform: {path}')
    time = (np.arange(8192) - p) / 50.0
    target, predicted = int(item['tgt_evt6']), int(item['pred_evt6'])
    station = item.get('station_uid', item.get('station_id_raw', 'unknown'))
    grid = outer[i // 2, i % 2].subgridspec(3, 1, hspace=.06)
    for channel, label in enumerate(('Z', 'N', 'E')):
        ax = fig.add_subplot(grid[channel, 0])
        ax.plot(time, wave[channel], color='#243c57', linewidth=.42)
        ax.axvline(0, color='#cb4335', linewidth=.8, linestyle='--')
        if np.isfinite(s) and p < s < 8192:
            ax.axvline((s - p) / 50, color='#2874a6', linewidth=.8, linestyle=':')
        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel(label, rotation=0, labelpad=12, va='center')
        ax.tick_params(axis='both', labelsize=7, length=2)
        ax.locator_params(axis='y', nbins=3)
        if channel == 0:
            ax.set_title(f'({chr(97+i)}) {names[target]} -> {names[predicted]} | event {item["event_uid"]} | station {station} | p={float(item[prob_cols[predicted]]):.3f}', loc='left', fontsize=9)
        if channel < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time relative to Pg (s)', fontsize=8)
    records.append({'panel': chr(97+i), 'target': names[target], 'prediction': names[predicted],
        'event_uid': item['event_uid'], 'station_uid': station, 'trace_uid': item['trace_uid'],
        'p_pick': p, 's_pick': s if np.isfinite(s) and p < s < 8192 else None,
        'predicted_probability': float(item[prob_cols[predicted]]),
        'waveform_path': str(path), 'waveform_sha256': sha(path)})
fig.suptitle('Correct and incorrect EVT6 test-waveform examples', fontsize=12)
fig.text(.075, .025, 'Each component is independently demeaned and divided by its standard deviation; '
         'vertical axes auto-scale. Red dashed: Pg; blue dotted: Sg when valid.', fontsize=8)
for extension in ('pdf', 'png'):
    fig.savefig(out / f'evt6_waveform_cases.{extension}', dpi=300, facecolor='white')
plt.close(fig)
pd.DataFrame(records).to_csv(out / 'selected_cases.csv', index=False)
config = {'seed': seed, 'selection': 'SHA-256 event-first selection; six distinct events',
    'normalization': 'per-component demean and standard deviation; vertical axes auto-scale',
    'sampling_rate_hz': 50, 'test_manifest_sha256': sha(manifest),
    'predictions_sha256': sha(prediction_file), 'script_sha256': sha(Path(__file__)),
    'note': 'Illustrative inputs and predictions; not an attribution or label verification.'}
(out / 'plot_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Results saved:', out, flush=True)
