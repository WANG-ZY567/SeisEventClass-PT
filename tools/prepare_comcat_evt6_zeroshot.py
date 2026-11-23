#!/usr/bin/env python3
"""Prepare a common ENZ-to-ZNE, phase-aligned ComCat cohort for EVT6 inference.

Only metadata-eligible eq/ep records are selected; no model predictions or
target labels are used to tune the preprocessing or model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

from pnw_ml_export_hdf5_to_npy import p_aligned_window, read_waveform_from_h5
from pnw_ml_prepare_comcat_for_diting_test import comcat_label, first_present


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    metadata = args.metadata.resolve(strict=True)
    hdf5 = args.hdf5.resolve(strict=True)
    out = args.output_dir.expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out = out.parent.resolve() / out.name

    records = []
    exclusions = Counter()
    event_labels = defaultdict(set)
    seen = set()
    with metadata.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = comcat_label(row)
            if label is None:
                exclusions['non_eq_ep_label'] += 1
                continue
            eid = first_present(row, ('event_id',))
            trace = first_present(row, ('trace_name',))
            station = first_present(row, ('station_code',))
            raw_p = finite(first_present(row, ('trace_P_arrival_sample', 'trace_p_arrival_sample', 'p_arrival_sample')))
            raw_s = finite(first_present(row, ('trace_S_arrival_sample', 'trace_s_arrival_sample', 's_arrival_sample')))
            rate = finite(first_present(row, ('trace_sampling_rate_hz', 'sampling_rate_hz')))
            if not eid or not trace or raw_p is None or raw_p < 0 or rate is None or rate <= 0:
                exclusions['missing_event_trace_p_or_rate'] += 1
                continue
            key = f'{eid}|{station}|{trace}'
            if key in seen:
                raise ValueError(f'Duplicate event/station/trace key: {key}')
            seen.add(key)
            event_labels[eid].add(label[0])
            uid = hashlib.sha256(key.encode('utf-8')).hexdigest()
            records.append({'key': key, 'part': 'test', 'trace_uid': uid,
                'event_uid': eid, 'comcat_event_id': eid, 'station_code': station,
                'trace_name': trace, '_npy_path': f'waves/{uid}.npy',
                '_evt6': label[0], '_evtype': ('eq', 'ep')[label[0]],
                'p_pick': 500, 's_pick': '', 'source_p_sample': raw_p,
                'source_s_sample': '' if raw_s is None else raw_s,
                'source_sampling_rate_hz': rate,
                'source_missing_channel': row.get('trace_missing_channel', ''),
                'padding_fraction_source_window': '', 'observed_post_p_seconds': ''})
    if not records:
        raise ValueError('No eligible earthquake/explosion records')
    conflicts = [event for event, labels in event_labels.items() if len(labels) != 1]
    if conflicts:
        raise ValueError(f'Conflicting eq/ep labels for {len(conflicts)} events, e.g. {conflicts[:5]}')
    required = len(records) * (3 * 8192 * 4 + 4096) + 1024**3
    free = shutil.disk_usage(out.parent).free
    if free < required:
        raise RuntimeError(f'Insufficient space for waveform cohort: need >= {required}, available={free}')

    out.mkdir()
    (out / 'waves').mkdir()
    sos = signal.butter(4, [1, 20], btype='bandpass', fs=50, output='sos')
    taper = signal.windows.tukey(8192, alpha=0.1)
    padding = []
    class_padding = defaultdict(list)
    missing_channels = Counter()
    missing_s = Counter()
    class_counts = Counter()
    with h5py.File(hdf5, 'r') as h5:
        if 'data_format/component_order' not in h5:
            raise ValueError('HDF5 has no data_format/component_order; component order cannot be verified')
        order = h5['data_format/component_order'][()]
        if isinstance(order, bytes):
            order = order.decode('ascii')
        if str(order).upper() != 'ENZ':
            raise ValueError(f'Expected ENZ input ordering, got {order!r}')
        for i, row in enumerate(records, 1):
            raw = np.asarray(read_waveform_from_h5(h5, row['trace_name']), dtype=np.float32)
            if raw.ndim != 2:
                raise ValueError(f'Non-2D waveform: {row["trace_name"]}: {raw.shape}')
            if raw.shape[0] == 3:
                pass
            elif raw.shape[1] == 3:
                raw = raw.T
            else:
                raise ValueError(f'Expected 3 components: {row["trace_name"]}: {raw.shape}')
            if not np.isfinite(raw).all():
                raise ValueError(f'Non-finite waveform: {row["trace_name"]}')
            rate, p0 = row['source_sampling_rate_hz'], row['source_p_sample']
            if not 0 <= p0 < raw.shape[1]:
                raise ValueError(f'P is outside waveform: {row["trace_name"]}')
            start = int(round(p0)) - int(round(10 * rate))
            total = int(round(163.84 * rate))
            end = start + total
            observed = max(0, min(end, raw.shape[1]) - max(start, 0))
            fraction = 1.0 - observed / total
            padding.append(fraction)
            class_padding[row['_evtype']].append(fraction)
            missing_channels[str(row['source_missing_channel'])] += 1
            row['padding_fraction_source_window'] = fraction
            row['observed_post_p_seconds'] = (raw.shape[1] - p0) / rate
            s0 = finite(row['source_s_sample'])
            if s0 is not None and p0 < s0 < raw.shape[1]:
                s = 500 + (s0 - p0) * 50 / rate
                before = int(round(0.05 * (s - 500)))
                length = min(150, before + int(round(0.60 * (s - 500))))
                valid_left = max(0.0, 500 - p0 * 50 / rate)
                valid_right = min(8192.0, 500 + (raw.shape[1] - p0) * 50 / rate)
                if 500 < s < 8192 and length > 0 and 500 - before >= valid_left and s - before + length <= valid_right:
                    row['s_pick'] = s
                else:
                    missing_s['s_or_ratio_window_not_fully_observed'] += 1
            else:
                missing_s['missing_invalid_or_outside_source_s'] += 1
            # The source declares E,N,Z; all three models use Z,N,E.
            aligned = p_aligned_window(raw[[2, 1, 0]], p0, rate, 50.0, 10.0, 153.84)
            if aligned.shape != (3, 8192):
                raise ValueError(f'Unexpected resampled shape: {aligned.shape}')
            x = signal.detrend(aligned.astype(np.float64), axis=1, type='linear')
            x = signal.sosfiltfilt(sos, x * taper[None, :], axis=1).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError(f'Non-finite processed waveform: {row["trace_name"]}')
            np.save(out / row['_npy_path'], x, allow_pickle=False)
            class_counts[row['_evtype']] += 1
            if i % 500 == 0 or i == len(records):
                print(f'[PREP] {i:,}/{len(records):,}', flush=True)

    fields = list(records[0])
    for split in ('train', 'val', 'test'):
        with (out / f'meta_evt6_{split}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            if split == 'test':
                writer.writerows(records)
    p = np.asarray(padding)
    report = {'source_metadata': str(metadata), 'source_metadata_sha256': digest(metadata),
        'source_hdf5': str(hdf5), 'source_hdf5_size_bytes': hdf5.stat().st_size,
        'source_component_order': 'ENZ', 'output_component_order': 'ZNE',
        'selection': 'eq/ep using project comcat_label: prefer PNSN eq/ep when present, otherwise source type; require event ID, trace name, P and rate',
        'records': len(records), 'events': len(event_labels), 'class_counts': dict(class_counts),
        'class_event_counts': {name: sum(labels == {index} for labels in event_labels.values())
                               for index, name in enumerate(('eq', 'ep'))},
        'excluded_metadata_counts': dict(exclusions), 'invalid_s_or_ratio_window_counts': dict(missing_s),
        'window': 'P-10 to P+153.84 s; zero pad missing samples before resampling and preprocessing',
        'padding_fraction_before_resampling_and_filtering': {'mean': float(p.mean()),
            'median': float(np.median(p)), 'max': float(p.max()),
            'records_with_padding': int(np.sum(p > 0))},
        'padding_fraction_by_class': {name: {'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'records': len(values)}
            for name, values in class_padding.items()},
        'source_missing_channel_values': dict(missing_channels),
        'preprocessing': 'resample to 50 Hz; linear detrend; Tukey alpha=0.1; 1-20 Hz order-4 zero-phase Butterworth; float32',
        'warning': 'Filtering and detrending can make originally zero-padded samples nonzero. Padding differs from the training data.',
        'manifest_sha256': digest(out / 'meta_evt6_test.csv')}
    (out / 'preparation_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
