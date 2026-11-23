"""Resplit an existing EVT6 cohort by raw event ID without copying waveforms."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--validation-fraction-of-development', type=float, default=0.1)
    args = p.parse_args()
    source = args.source_dir.resolve()
    output = args.out_dir.resolve()
    if source == output:
        raise ValueError('source and output directories must differ')
    if not 0 < args.validation_fraction_of_development < 1:
        raise ValueError('validation fraction must be between 0 and 1')
    manifest = source / 'meta_evt6.csv'
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {'event_id_raw', '_rid', '_evt6', '_npy_path', 'split'}
    if not required.issubset(df.columns):
        raise ValueError(f'missing required columns: {sorted(required - set(df.columns))}')
    for col in required:
        df[col] = df[col].str.strip()
        if df[col].eq('').any():
            raise ValueError(f'{col} contains empty values')
    df['event_id_raw'] = df['event_id_raw'].str.replace(r'\.0$', '', regex=True)
    if not df['event_id_raw'].str.fullmatch(r'\d+').all():
        raise ValueError('event_id_raw must be a non-negative integer')
    if df['_rid'].duplicated().any():
        raise ValueError('duplicate raw record IDs found; review before splitting')
    if (df['event_id_raw'] != df['_rid'].str.rsplit('_', n=1).str[0]).any():
        raise ValueError('event_id_raw does not match record ID prefix')
    event_labels = df.groupby('event_id_raw')['_evt6'].nunique()
    if event_labels.gt(1).any():
        raise ValueError(f'raw events with multiple labels: {event_labels[event_labels.gt(1)].index[:10].tolist()}')
    event_label = df.drop_duplicates('event_id_raw')[['event_id_raw', '_evt6']]
    assignment = {}
    allocations = {}
    for label, group in event_label.groupby('_evt6', sort=True):
        events = sorted(group['event_id_raw'].tolist(), key=int)
        count = len(events)
        if count < 3:
            raise ValueError(f'class {label} has fewer than three events')
        n_test = max(1, round(count * 0.2))
        n_dev = count - n_test
        n_val = max(1, round(n_dev * args.validation_fraction_of_development))
        if n_dev - n_val < 1:
            raise ValueError(f'class {label} has too few training events')
        n_train = n_dev - n_val
        for event in events[:n_train]: assignment[event] = 'train'
        for event in events[n_train:n_dev]: assignment[event] = 'val'
        for event in events[n_dev:]: assignment[event] = 'test'
        allocations[label] = {'events_total': count, 'train': n_train, 'val': n_val, 'test': n_test}
    previous = df['split'].copy()
    df['split'] = df['event_id_raw'].map(assignment)
    paths = []
    for value in df['_npy_path']:
        path = Path(value)
        path = path if path.is_absolute() else source / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'waveform does not exist: {path}')
        paths.append(str(path))
    df['_npy_path'] = paths
    for name in ('meta_evt6.csv', 'meta_evt6_train.csv', 'meta_evt6_val.csv', 'meta_evt6_test.csv', 'resplit_report.json'):
        if (output / name).exists():
            raise FileExistsError(f'output exists: {output / name}')
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths = [output / 'meta_evt6.csv']
    df.to_csv(manifest_paths[0], index=False)
    summary = {}
    for split in ('train', 'val', 'test'):
        part = df[df['split'] == split]
        path = output / f'meta_evt6_{split}.csv'
        part.to_csv(path, index=False)
        manifest_paths.append(path)
        summary[split] = {'records': len(part), 'events': part['event_id_raw'].nunique(),
                          'records_by_class': dict(Counter(part['_evt6'])),
                          'events_by_class': part.groupby('_evt6')['event_id_raw'].nunique().to_dict()}
    sets = {s: set(df.loc[df['split'] == s, 'event_id_raw']) for s in ('train', 'val', 'test')}
    overlap = {f'{a}_{b}': len(sets[a] & sets[b]) for a, b in (('train','val'), ('train','test'), ('val','test'))}
    if any(overlap.values()):
        raise AssertionError(f'event overlap: {overlap}')
    report = {'source_dir': str(source), 'source_manifest_sha256': digest(manifest),
              'event_sort': 'numeric event_id_raw within class (catalogue-order proxy; not verified origin time)',
              'test_fraction_of_events': 0.2,
              'validation_fraction_of_development_events': args.validation_fraction_of_development,
              'allocation': allocations, 'splits': summary, 'event_overlap_counts': overlap,
              'old_split_to_new_split_records': pd.crosstab(previous, df['split']).to_dict(),
              'waveforms': 'absolute paths to original filtered waveforms; source directory must remain accessible',
              'manifests_sha256': {p.name: digest(p) for p in manifest_paths}}
    (output / 'resplit_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
