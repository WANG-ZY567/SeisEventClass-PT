"""Event-grouped source probe using paired pre-P waveforms from training events."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def norm_id(value):
    value = str(value).strip()
    return value[:-2] if value.endswith('.0') else value


def crop(h5, info, rid):
    pg, sg = float(info['Pg']), float(info['Sg'])
    if not (math.isfinite(pg) and math.isfinite(sg) and sg > pg):
        raise ValueError('invalid picks')
    wave = np.asarray(h5[rid][()])
    if wave.ndim != 2:
        raise ValueError('invalid shape')
    if wave.shape[1] != 3 and wave.shape[0] == 3:
        wave = wave.T
    if wave.shape[1] != 3:
        raise ValueError('invalid channels')
    p = int(np.floor(pg))
    start, end = p - 500, p + 7692
    if start < 0 or end > len(wave) or sg + 2000 > end:
        raise ValueError('window out of bounds')
    segment = wave[start:end].astype(np.float64)
    if not np.isfinite(segment).all() or np.any(np.ptp(segment, axis=0) == 0):
        raise ValueError('invalid waveform')
    return segment


def features(segment):
    # Filter only P-10 to P-2; using no post-P samples avoids future leakage.
    x = segment[:400].T
    x = signal.detrend(x, axis=1, type='linear')
    x *= signal.windows.tukey(x.shape[1], alpha=0.1)[None, :]
    sos = signal.butter(4, [1, 20], btype='bandpass', fs=50, output='sos')
    x = signal.sosfiltfilt(sos, x, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    if np.any(std < 1e-12):
        raise ValueError('constant pre-P component')
    x /= std
    result = []
    for c in x:
        freqs, power = signal.welch(c, fs=50, nperseg=256)
        total = power.sum() + 1e-20
        for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 15), (15, 20)):
            result.append(float(power[(freqs >= lo) & (freqs < hi)].sum() / total))
        result.extend([float(stats.skew(c)), float(stats.kurtosis(c))])
        for lag in (1, 2, 5, 10):
            result.append(float(np.dot(c[:-lag], c[lag:]) / np.dot(c, c)))
    corr = np.corrcoef(x)
    result.extend([float(corr[0, 1]), float(corr[0, 2]), float(corr[1, 2])])
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError('invalid features')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    args = parser.parse_args()
    dfs = {s: pd.read_csv(args.data_dir / f'meta_evt6_{s}.csv', dtype=str)
           for s in ('train', 'val', 'test')}
    heldout = {norm_id(x) for s in ('val', 'test') for x in dfs[s]['event_id_raw'].dropna()}
    train = dfs['train'][dfs['train']['_src'] == 'non']
    paths = {
        'natural': ('natural_readable_3c_nonzero_v1.json', 'CENC_DiTingv2_natural_earthquake.hdf5'),
        'non': ('non_natural_readable_3c_nonzero_v1.json', 'CENC_DiTingv2_non_natural_earthquake.hdf5'),
    }
    meta = {}
    for source, (name, _) in paths.items():
        with (args.raw_dir / name).open(encoding='utf-8') as f:
            meta[source] = json.load(f)
    X, y, groups, classes = [], [], [], []
    counts, failures = Counter(), Counter()
    with h5py.File(args.raw_dir / paths['natural'][1], 'r') as ha, h5py.File(args.raw_dir / paths['non'][1], 'r') as hb:
        for rid in sorted(set(train['_rid'].dropna().astype(str))):
            if rid not in meta['natural'] or rid not in meta['non']:
                continue
            event = norm_id(rid.rsplit('_', 1)[0])
            if event in heldout:
                continue
            a, b = meta['natural'][rid], meta['non'][rid]
            cls = str(a.get('evtype', '')).strip().lower()
            if cls not in ('ep', 'ss', 'sp') or cls != str(b.get('evtype', '')).strip().lower():
                continue
            try:
                fa, fb = features(crop(ha, a, rid)), features(crop(hb, b, rid))
            except (KeyError, ValueError, TypeError, OSError) as exc:
                failures[type(exc).__name__] += 1
                continue
            X.extend((fa, fb)); y.extend((0, 1)); groups.extend((event, event)); classes.extend((cls, cls))
            counts[cls] += 1
    X, y, groups, classes = map(np.asarray, (X, y, groups, classes))
    if not len(y) or len(set(groups)) < 5:
        raise ValueError('insufficient paired events')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261112)
    folds = []
    for number, (tr, te) in enumerate(cv.split(X, classes, groups), 1):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20261112))
        model.fit(X[tr], y[tr])
        score = model.predict_proba(X[te])[:, 1]
        folds.append({'fold': number, 'train_events': len(set(groups[tr])), 'test_events': len(set(groups[te])),
                      'pairs': int(len(te) // 2), 'auc': float(roc_auc_score(y[te], score)),
                      'accuracy': float(accuracy_score(y[te], score >= 0.5))})
    report = {'scope': 'paired training events, ep/ss/sp, P-10 to P-2 samples; filter and normalize pre-P alone',
              'sampling_rate_assumed_hz': 50, 'pairs_by_class': dict(counts), 'events': len(set(groups)),
              'failures': dict(failures), 'folds': folds,
              'mean_auc': float(np.mean([f['auc'] for f in folds])),
              'mean_accuracy': float(np.mean([f['accuracy'] for f in folds]))}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
