"""Exploratory event-grouped source probe on non-overlapping records."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_diting2_paired_source import crop, features, norm_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    args = p.parse_args()
    paths = {
        'natural': ('natural_readable_3c_nonzero_v1.json', 'CENC_DiTingv2_natural_earthquake.hdf5'),
        'non': ('non_natural_readable_3c_nonzero_v1.json', 'CENC_DiTingv2_non_natural_earthquake.hdf5'),
    }
    meta = {}
    for src, (name, _) in paths.items():
        with (args.raw_dir / name).open(encoding='utf-8') as f:
            meta[src] = json.load(f)
    dfs = {s: pd.read_csv(args.data_dir / f'meta_evt6_{s}.csv', dtype=str)
           for s in ('train', 'val', 'test')}
    used = {norm_id(x) for df in dfs.values() for x in df['event_id_raw'].dropna()}
    heldout = {norm_id(x) for s in ('val', 'test') for x in dfs[s]['event_id_raw'].dropna()}
    train_non = set(dfs['train'].loc[dfs['train']['_src'] == 'non', '_rid'].dropna().astype(str))
    labels = ('ep', 'ss', 'sp')
    candidates = {'natural': [], 'non': []}
    for rid, info in meta['natural'].items():
        cls = str(info.get('evtype', '')).strip().lower()
        event = norm_id(str(rid).rsplit('_', 1)[0])
        if cls in labels and rid not in meta['non'] and event not in used:
            candidates['natural'].append((rid, event, cls))
    for rid in train_non:
        if rid in meta['natural'] or rid not in meta['non']:
            continue
        event = norm_id(rid.rsplit('_', 1)[0])
        cls = str(meta['non'][rid].get('evtype', '')).strip().lower()
        if cls in labels and event not in heldout:
            candidates['non'].append((rid, event, cls))
    X, ys, groups, classes, sp = [], [], [], [], []
    skipped = Counter()
    event_strata = {}
    for source in ('natural', 'non'):
        with h5py.File(args.raw_dir / paths[source][1], 'r') as h5:
            for rid, event, cls in sorted(candidates[source]):
                stratum = (source, cls)
                if event in event_strata and event_strata[event] != stratum:
                    raise ValueError(f'event {event} has inconsistent source/class')
                try:
                    info = meta[source][rid]
                    f = features(crop(h5, info, rid))
                    delta = (float(info['Sg']) - float(info['Pg'])) / 50.0
                except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
                    skipped[type(exc).__name__] += 1
                    continue
                event_strata[event] = stratum
                X.append(f); ys.append(int(source == 'non')); groups.append(event); classes.append(cls); sp.append(delta)
    X, y, groups, classes, sp = map(np.asarray, (X, ys, groups, classes, sp))
    events = np.asarray(sorted(event_strata))
    strata = np.asarray([f'{event_strata[e][0]}:{event_strata[e][1]}' for e in events])
    event_counts = Counter(strata)
    if any(event_counts.get(f'{src}:{cls}', 0) < 5 for src in paths for cls in labels):
        raise ValueError(f'Need at least 5 events per source/class: {dict(event_counts)}')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261112)
    folds = []
    for number, (tr_e, te_e) in enumerate(cv.split(events, strata), 1):
        tr = np.flatnonzero(np.isin(groups, events[tr_e])); te = np.flatnonzero(np.isin(groups, events[te_e]))
        # Equal total contribution per event within each source/class stratum.
        event_n = Counter(groups[tr]); stratum_n = Counter(strata[tr_e])
        weights = np.asarray([1.0 / (event_n[groups[i]] * stratum_n[f'{"non" if y[i] else "natural"}:{classes[i]}']) for i in tr])
        weights *= len(weights) / weights.sum()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20261112))
        model.fit(X[tr], y[tr], logisticregression__sample_weight=weights)
        scores = model.predict_proba(X[te])[:, 1]
        event_y, event_cls, event_score = [], [], []
        for event in events[te_e]:
            mask = groups[te] == event
            event_y.append(int(y[te][mask][0])); event_cls.append(classes[te][mask][0]); event_score.append(float(scores[mask].mean()))
        event_y, event_cls, event_score = map(np.asarray, (event_y, event_cls, event_score))
        aucs = {cls: float(roc_auc_score(event_y[event_cls == cls], event_score[event_cls == cls])) for cls in labels}
        folds.append({'fold': number, 'test_events': len(te_e), 'auc_by_class': aucs,
                      'macro_class_auc': float(np.mean(list(aucs.values()))),
                      'balanced_accuracy': float(balanced_accuracy_score(event_y, event_score >= .5))})
    diagnostics = {}
    for src, code in [('natural', 0), ('non', 1)]:
        for cls in labels:
            mask = (y == code) & (classes == cls)
            diagnostics[f'{src}:{cls}'] = {'traces': int(mask.sum()), 'events': len(set(groups[mask])),
                                            'sp_q10_median_q90_seconds': np.quantile(sp[mask], [.1, .5, .9]).round(3).tolist()}
    report = {'scope': 'distinct records and raw event IDs; main test/validation events excluded; ep/ss/sp only',
              'features': 'pre-P samples 0:400; detrend, local band-pass and per-component local normalization',
              'limitations': 'unmatched events/stations/time; source and event population may be confounded',
              'sampling_rate_assumed_hz': 50, 'diagnostics': diagnostics, 'skipped': dict(skipped),
              'folds': folds, 'mean_macro_class_auc': float(np.mean([f['macro_class_auc'] for f in folds]))}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
