#!/usr/bin/env python3
"""Package only the reproducibly selected EVT6 waveform examples for plotting elsewhere."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import plot_evt6_waveform_cases as plotter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    data = args.data_dir.resolve(strict=True)
    predictions = args.predictions.resolve(strict=True)
    manifest = data / 'meta_evt6_test.csv'
    output = args.output_dir.expanduser().absolute()
    archive = args.archive.expanduser().absolute()
    for path in (output, archive):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite: {path}')
    meta = pd.read_csv(manifest, dtype={'trace_uid': str, 'event_uid': str}, low_memory=False)
    pred = pd.read_csv(predictions, dtype={'trace_uid': str, 'event_uid': str}, low_memory=False)
    for name, frame, required in (
        ('manifest', meta, {'trace_uid', 'event_uid', '_npy_path', '_evt6', 'p_pick', 's_pick'}),
        ('predictions', pred, {'trace_uid', 'tgt_evt6', 'pred_evt6'}),
    ):
        if required - set(frame.columns):
            raise ValueError(f'{name} missing columns: {sorted(required - set(frame.columns))}')
        if frame['trace_uid'].isna().any() or frame['trace_uid'].duplicated().any():
            raise ValueError(f'{name} contains missing or duplicate trace identifiers')
    if meta['event_uid'].isna().any() or set(meta['trace_uid']) != set(pred['trace_uid']):
        raise ValueError('Missing event IDs or predictions do not match the test manifest')
    frame = meta.merge(pred[['trace_uid', 'tgt_evt6', 'pred_evt6']], on='trace_uid', validate='one_to_one')
    for col in ('_evt6', 'tgt_evt6', 'pred_evt6'):
        values = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or not values.between(0, 5).all():
            raise ValueError(f'Invalid labels: {col}')
        frame[col] = values.astype(int)
    if not (frame['_evt6'] == frame['tgt_evt6']).all():
        raise ValueError('Target labels differ between manifest and predictions')
    selected, counts = plotter.select_cases(frame, args.seed)
    sources = {}
    for row in selected:
        source = Path(str(row['_npy_path']).strip())
        source = (source if source.is_absolute() else data / source).resolve(strict=True)
        wave = np.load(source, allow_pickle=False)
        if wave.shape != (3, 8192) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid waveform: {source} {wave.shape}')
        p = plotter.finite_number(row['p_pick'])
        if p is None or not 0 <= p < 8192:
            raise ValueError(f'Invalid P pick for {row["trace_uid"]}')
        sources[row['trace_uid']] = source
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    (output / 'waves').mkdir()
    # Keep every metadata and prediction row to preserve and verify selection;
    # only the six selected waveforms are copied. This bundle is plot-only.
    source_hashes = {}
    for uid, source in sources.items():
        filename = f'{plotter.rank(0, "file", uid)}.npy'
        destination = output / 'waves' / filename
        shutil.copy2(source, destination)
        source_hashes[uid] = {'source': str(source), 'source_sha256': plotter.sha256(source),
                              'bundled_path': f'waves/{filename}'}
    relative = {uid: item['bundled_path'] for uid, item in source_hashes.items()}
    meta['_npy_path'] = meta['trace_uid'].map(lambda uid: relative.get(uid, 'not_in_plot_bundle.npy'))
    meta.to_csv(output / 'meta_evt6_test.csv', index=False)
    shutil.copy2(predictions, output / 'predictions.csv')
    shutil.copy2(Path(plotter.__file__).resolve(), output / 'plot_evt6_waveform_cases.py')
    info = {'selection_seed': args.seed, 'source_data': str(data),
        'source_manifest_sha256': plotter.sha256(manifest),
        'source_predictions_sha256': plotter.sha256(predictions),
        'plot_script_sha256': plotter.sha256(Path(plotter.__file__).resolve()),
        'test_manifest_records': len(meta), 'waveforms_in_bundle': len(sources),
        'warning': 'Plot-only bundle: all test metadata retained but only selected waveform files are included.',
        'candidate_counts': counts, 'selected_waveforms': source_hashes}
    (output / 'bundle_provenance.json').write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with tarfile.open(archive, mode='x:gz') as tar:
        tar.add(output, arcname='evt6_waveform_plot_bundle', recursive=True)
    print('Bundle archive:', archive)
    print('Archive SHA-256:', plotter.sha256(archive))
    print('Archive size:', archive.stat().st_size, 'bytes')
    print('Test metadata rows:', len(meta), '; selected waveform files:', len(sources))
    for row in selected:
        print(f"{plotter.CLASSES[int(row['tgt_evt6'])]} -> {plotter.CLASSES[int(row['pred_evt6'])]}: event={row['event_uid']} trace={row['trace_uid']}")


if __name__ == '__main__':
    main()
