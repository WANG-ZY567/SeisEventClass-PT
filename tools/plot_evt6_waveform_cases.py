#!/usr/bin/env python3
"""Select and plot reproducible correct/error EVT6 test-waveform examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ('eq', 'ep', 'ss', 'sp', 'se', 'ot')
CASES = ((3, 3), (3, 0), (4, 4), (4, 0), (5, 5), (5, 2))
CHANNELS = ('Z', 'N', 'E')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def rank(seed, purpose, value):
    return hashlib.sha256(f'{seed}|{purpose}|{value}'.encode('utf-8')).hexdigest()


def select_cases(frame, seed):
    groups, counts = [], []
    for target, predicted in CASES:
        part = frame[(frame['tgt_evt6'] == target) & (frame['pred_evt6'] == predicted)].copy()
        if part.empty:
            raise ValueError(f'No test records for {CLASSES[target]} -> {CLASSES[predicted]}')
        events = sorted(part['event_uid'].unique(), key=lambda e: rank(seed, f'{target}-{predicted}-event', e))
        groups.append((target, predicted, part, events))
        counts.append({'target': CLASSES[target], 'predicted': CLASSES[predicted],
                       'candidate_waveforms': len(part), 'candidate_events': len(events)})

    # Backtrack if needed to ensure all six examples are from distinct events.
    def solve(index, used):
        if index == len(groups):
            return []
        target, predicted, _, events = groups[index]
        for event in events:
            if event in used:
                continue
            rest = solve(index + 1, used | {event})
            if rest is not None:
                return [event] + rest
        return None

    chosen_events = solve(0, set())
    if chosen_events is None:
        raise ValueError('Cannot select the six requested cases from distinct events')
    chosen = []
    for (target, predicted, part, _), event in zip(groups, chosen_events):
        subset = part[part['event_uid'] == event].copy()
        ordered = sorted(subset['trace_uid'], key=lambda uid: rank(seed, f'{target}-{predicted}-trace', uid))
        chosen.append(subset.set_index('trace_uid').loc[ordered[0]].to_dict() | {'trace_uid': ordered[0]})
    return chosen, counts


def finite_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--sampling-rate', type=float, default=50.0)
    args = parser.parse_args()
    if args.sampling_rate <= 0:
        raise ValueError('Sampling rate must be positive')
    data = args.data_dir.expanduser().resolve(strict=True)
    predictions_path = args.predictions.expanduser().resolve(strict=True)
    manifest = data / 'meta_evt6_test.csv'
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output directory: {output}')
    meta = pd.read_csv(manifest, dtype={'trace_uid': str, 'event_uid': str}, low_memory=False)
    pred = pd.read_csv(predictions_path, dtype={'trace_uid': str, 'event_uid': str}, low_memory=False)
    for name, frame, required in (
        ('manifest', meta, {'trace_uid', 'event_uid', '_npy_path', '_evt6', 'p_pick', 's_pick'}),
        ('predictions', pred, {'trace_uid', 'tgt_evt6', 'pred_evt6'}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name} missing columns: {sorted(missing)}')
        if frame['trace_uid'].isna().any() or frame['trace_uid'].duplicated().any():
            raise ValueError(f'{name}: missing or duplicate trace identifiers')
    if meta['event_uid'].isna().any():
        raise ValueError('Missing event identifiers in test manifest')
    if set(meta['trace_uid']) != set(pred['trace_uid']):
        raise ValueError('Predictions and test manifest do not contain the same records')
    # Manifest is authoritative for waveform paths, picks, and catalogue identifiers.
    pred_cols = ['trace_uid', 'tgt_evt6', 'pred_evt6']
    prob_cols = [f'prob_evt6_{i}' for i in range(6)]
    if all(c in pred.columns for c in prob_cols):
        pred_cols += prob_cols
    elif any(c in pred.columns for c in prob_cols):
        raise ValueError('Only some of the six probability columns are present')
    frame = meta.merge(pred[pred_cols], on='trace_uid', how='inner', validate='one_to_one')
    for col in ('_evt6', 'tgt_evt6', 'pred_evt6'):
        values = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or not values.between(0, 5).all():
            raise ValueError(f'Invalid six-class values: {col}')
        frame[col] = values.astype(int)
    if not (frame['_evt6'] == frame['tgt_evt6']).all():
        raise ValueError('Test manifest and predictions disagree on target labels')
    if prob_cols[0] in frame.columns:
        values = frame[prob_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < -1e-6) or np.any(values > 1 + 1e-6):
            raise ValueError('Invalid prediction probabilities')
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError('Prediction probabilities do not sum to one')
        if not (values.argmax(axis=1) == frame['pred_evt6'].to_numpy()).all():
            raise ValueError('Saved predictions disagree with probability argmax')

    selected, candidate_counts = select_cases(frame, args.seed)
    loaded = []
    for item in selected:
        wave_path = Path(str(item['_npy_path']).strip())
        if not wave_path.is_absolute():
            wave_path = data / wave_path
        wave_path = wave_path.resolve(strict=True)
        wave = np.load(wave_path, allow_pickle=False).astype(np.float32, copy=True)
        if wave.shape != (3, 8192) or not np.isfinite(wave).all():
            raise ValueError(f'Expected finite waveform [3,8192]: {wave_path} {wave.shape}')
        p = finite_number(item['p_pick'])
        s = finite_number(item['s_pick'])
        if p is None or not 0 <= p < wave.shape[1]:
            raise ValueError(f'Valid P pick required for plotting: {item["trace_uid"]}')
        if s is None or not p < s < wave.shape[1]:
            s = None
        # Reproduce the test-time per-component mean/std normalization.
        wave -= wave.mean(axis=1, keepdims=True)
        std = wave.std(axis=1, keepdims=True)
        std[std == 0] = 1
        wave /= std
        if not np.isfinite(wave).all():
            raise ValueError(f'Non-finite normalized waveform: {wave_path}')
        item['resolved_waveform_path'] = str(wave_path)
        item['waveform_sha256'] = sha256(wave_path)
        item['displayed_p_sample'] = p
        item['displayed_s_sample'] = '' if s is None else s
        loaded.append((wave, p, s))

    output.mkdir(parents=True, exist_ok=False)
    matplotlib.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(13.0, 10.6), constrained_layout=False)
    outer = fig.add_gridspec(3, 2, left=.075, right=.985, bottom=.075, top=.925, wspace=.24, hspace=.48)
    for index, (item, (wave, p, s)) in enumerate(zip(selected, loaded)):
        row, col = divmod(index, 2)
        grid = outer[row, col].subgridspec(3, 1, hspace=.06)
        time = (np.arange(wave.shape[1]) - p) / args.sampling_rate
        target, predicted = int(item['tgt_evt6']), int(item['pred_evt6'])
        station = str(item.get('station_uid', item.get('station_id_raw', 'unknown')))
        title = f"({chr(97 + index)}) {CLASSES[target]} -> {CLASSES[predicted]} | event {item['event_uid']} | station {station}"
        if prob_cols[0] in frame.columns:
            title += f" | p(pred)={float(item[prob_cols[predicted]]):.3f}"
        for channel in range(3):
            ax = fig.add_subplot(grid[channel, 0])
            ax.plot(time, wave[channel], color='#243c57', linewidth=.42)
            ax.axvline(0, color='#cb4335', linewidth=.8, linestyle='--')
            if s is not None:
                ax.axvline((s - p) / args.sampling_rate, color='#2874a6', linewidth=.8, linestyle=':')
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel(CHANNELS[channel], rotation=0, labelpad=12, va='center')
            ax.tick_params(axis='both', labelsize=7, length=2)
            ax.locator_params(axis='y', nbins=3)
            if channel == 0:
                ax.set_title(title, loc='left', fontsize=9, pad=5)
            if channel < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time relative to Pg (s)', fontsize=8)
    fig.suptitle('Correct and incorrect EVT6 test-waveform examples', fontsize=12)
    fig.text(.075, .025, 'Each component is independently demeaned and divided by its standard deviation; '
             'vertical axes are automatically scaled. Red dashed: Pg; blue dotted: Sg when valid.', fontsize=8)
    for extension in ('pdf', 'png'):
        fig.savefig(output / f'evt6_waveform_cases.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    case_frame = pd.DataFrame(selected)
    case_frame.insert(0, 'panel', [chr(97+i) for i in range(len(selected))])
    case_frame.to_csv(output / 'selected_cases.csv', index=False)
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, output / 'plot_script.py')
    report = {'selection_seed': args.seed, 'selection': 'SHA-256 ordering; choose events first, then a waveform; six distinct events',
              'candidate_counts': candidate_counts, 'sampling_rate_hz': args.sampling_rate,
              'normalization': 'per-component demean and population standard deviation; zero std uses 1',
              'display': 'full 8192 samples; time relative to Pg; each axes auto-scales',
              'interpretation': 'Waveform examples describe inputs and predictions; they do not establish model attribution or label errors.',
              'plot_script_sha256': sha256(script_path),
              'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
              'predictions': str(predictions_path), 'predictions_sha256': sha256(predictions_path)}
    (output / 'plot_config.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for row in candidate_counts:
        print(f"{row['target']} -> {row['predicted']}: {row['candidate_waveforms']} waveforms, {row['candidate_events']} events")
    print(f'Figure and selection records saved: {output}')


if __name__ == '__main__':
    main()
