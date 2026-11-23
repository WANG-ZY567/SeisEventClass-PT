"""Add unique non-eq natural-file records to an existing filtered EVT6 cohort."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from build_diting2_evt6_review_protocol import (
    LABEL_TO_ID, OUTPUT_COLUMNS, _load_json, _normalise_label, _split_record_id,
    _to_float, _qc_candidates, _read_three_component, _atomic_save_npy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--natural-json', required=True, type=Path)
    parser.add_argument('--non-json', required=True, type=Path)
    parser.add_argument('--natural-h5', required=True, type=Path)
    parser.add_argument('--non-h5', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=('plan', 'build'), required=True)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.out_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError('output must not be the source directory or inside it')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output directory must be empty: {output}')
    base = pd.read_csv(source / 'meta_evt6.csv', dtype=str, keep_default_na=False)
    if set(OUTPUT_COLUMNS) - set(base.columns):
        raise ValueError('source manifest lacks required columns')
    if base['_rid'].duplicated().any():
        raise ValueError('source cohort contains duplicate raw record IDs')
    for col in ('Pg', 'Sg'):
        base[col] = pd.to_numeric(base[col], errors='coerce')
    valid = base['Pg'].notna() & base['Sg'].notna() & (base['Sg'] > base['Pg'])
    invalid_base = int((~valid).sum())
    base = base.loc[valid].copy()
    base['event_id_raw'] = base['event_id_raw'].astype(str).str.replace(r'\.0$', '', regex=True)
    if not base['event_id_raw'].str.fullmatch(r'\d+').all():
        raise ValueError('invalid raw event ID in source cohort')
    event_labels = {}
    for event, cls in zip(base['event_id_raw'], base['_evtype']):
        old = event_labels.setdefault(event, cls)
        if old != cls:
            raise ValueError(f'raw event {event} has mixed source labels: {old}, {cls}')
    for col in ('Pg', 'Sg'):
        base[col] = base[col].astype(str)
    base['_npy_path'] = base['_npy_path'].map(lambda x: str((source / x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()))
    missing = next((x for x in base['_npy_path'] if not Path(x).is_file()), None)
    if missing:
        raise FileNotFoundError(f'missing source waveform: {missing}')

    print('[LOAD] metadata', flush=True)
    natural, non = _load_json(args.natural_json), _load_json(args.non_json)
    event_classes = defaultdict(set)
    for rid, info in natural.items():
        cls = _normalise_label(info.get('evtype'))
        if cls in LABEL_TO_ID:
            event, _ = _split_record_id(str(rid))
            event_classes[event].add(cls)
    mixed = {event for event, labels in event_classes.items() if len(labels) > 1}
    non_event_classes = defaultdict(set)
    for rid, info in non.items():
        cls = _normalise_label(info.get('evtype'))
        if cls in LABEL_TO_ID:
            event, _ = _split_record_id(str(rid))
            non_event_classes[event].add(cls)
    base_ids = set(base['_rid'])
    counts, candidates = Counter(), []
    for rid, info in natural.items():
        rid = str(rid)
        cls = _normalise_label(info.get('evtype'))
        if cls not in ('ep', 'ss', 'sp', 'se', 'ot'):
            continue
        event, station = _split_record_id(rid)
        reason = None
        if not event.isdigit():
            reason = 'invalid_raw_event_id'
        elif rid in base_ids:
            reason = 'duplicate_record_id'
        elif event in mixed:
            reason = 'mixed_label_event'
        elif event in non_event_classes and non_event_classes[event] != {cls}:
            reason = 'conflict_with_non_file_event_label'
        elif event in event_labels and event_labels[event] != cls:
            reason = 'conflict_with_existing_event_label'
        else:
            pg, sg = _to_float(info.get('Pg')), _to_float(info.get('Sg'))
            if pg is None or sg is None or sg <= pg:
                reason = 'missing_or_invalid_phase_order'
        if reason:
            counts[f'{cls}:{reason}'] += 1
            continue
        candidates.append({'source': 'natural', 'record_id': rid, 'event_id': event,
                           'station_id': station, 'label': cls, 'meta': info})
    del non
    print(f'[QC] natural non-eq candidates: {len(candidates)}', flush=True)
    added, excluded, reasons = _qc_candidates(candidates, args.natural_h5, args.non_h5,
                                                8192, 500, True, 2000)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'build':
        waves = output / 'waves'
        waves.mkdir()
        sos = signal.butter(4, [1, 20], btype='bandpass', fs=50, output='sos')
        taper = signal.windows.tukey(8192, alpha=0.1)
        with h5py.File(args.natural_h5, 'r') as h5:
            for index, row in enumerate(added, 1):
                wave = _read_three_component(h5, row['_rid'])
                start = int(round(float(row['Pg']))) - 500
                window = wave[:, start:start + 8192].astype(np.float32)
                if window.shape != (3, 8192) or not np.isfinite(window).all():
                    raise ValueError(f"QC/export mismatch: {row['_rid']}")
                # Match create_evt6_bandpass_dataset.py: raw crop saved as float32 first.
                x = signal.detrend(window.astype(np.float64), axis=1, type='linear')
                filtered = signal.sosfiltfilt(sos, x * taper[None, :], axis=1)
                if not np.isfinite(filtered).all():
                    raise ValueError(f"filter failed: {row['_rid']}")
                _atomic_save_npy(waves / Path(row['_npy_path']).name, filtered.astype(np.float32))
                if index % 500 == 0 or index == len(added):
                    print(f'[EXPORT] {index}/{len(added)}', flush=True)
    for row in added:
        row['split'] = 'unassigned'
    combined = pd.concat([base, pd.DataFrame(added, columns=OUTPUT_COLUMNS)], ignore_index=True)
    combined['event_uid'] = 'event_' + combined['event_id_raw'].astype(str)
    if combined['_rid'].duplicated().any():
        raise AssertionError('duplicate record IDs in combined cohort')
    mixed_combined = combined.groupby('event_id_raw')['_evtype'].nunique()
    if mixed_combined.gt(1).any():
        raise AssertionError('mixed labels in combined cohort')
    combined.to_csv(output / 'meta_evt6.csv', index=False)
    pd.DataFrame(excluded, columns=['source','record_id','event_id','station_id','label','reason','detail']).to_csv(output / 'qc_excluded.csv', index=False)
    report = {'mode': args.mode, 'source_dir': str(source), 'base_records_after_phase_order_qc': len(base),
              'base_records_excluded_invalid_phase_order': invalid_base,
              'duplicate_policy': 'keep existing cohort version for duplicate raw record IDs; natural version may be used when non version was not selected',
              'added_records': len(added), 'added_events': len({row['event_id_raw'] for row in added}),
              'added_records_by_class': dict(Counter(row['_evtype'] for row in added)),
              'candidate_exclusions': dict(counts), 'qc_exclusions': dict(reasons),
              'combined_records': len(combined), 'combined_raw_events': combined['event_id_raw'].nunique(),
              'records_by_class_and_source': pd.crosstab(combined['_evtype'], combined['_src']).to_dict(),
              'preprocessing': '8192 samples, Pg at 500, Sg after Pg and >=2000 samples remain; 50 Hz, detrend, Tukey 0.05 each edge, 1-20 Hz order-4 zero-phase Butterworth',
              'waveforms': 'base records reference source directory; new records saved in output/waves',
              'note': 'plan mode does not export waveforms; build mode must run before resplitting'}
    (output / 'augmentation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
