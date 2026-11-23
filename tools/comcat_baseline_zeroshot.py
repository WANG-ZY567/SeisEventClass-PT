#!/usr/bin/env python3
"""Prepare EQTypeNet external features and infer fixed EQTypeNet/CapsNet models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ensure_new(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    path.mkdir(parents=True)


def prepare_eq(args):
    data, output = args.data.resolve(strict=True), args.output.absolute()
    train_report = json.loads(args.train_report.read_text(encoding='utf-8'))
    fallback = float(train_report['training_event_ratio_median'])
    if not np.isfinite(fallback):
        raise ValueError('Training-set event median is not finite')
    frame = pd.read_csv(data / 'meta_evt6_test.csv', low_memory=False)
    if frame.empty or frame['trace_uid'].duplicated().any():
        raise ValueError('External cohort is empty or contains duplicate trace identifiers')
    ensure_new(output)
    mod = load_module('eqt_external_preparation', args.eq_prepare.resolve(strict=True))
    tasks = ((row, data, output, 'test') for row in frame.to_dict('records'))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            features = list(tqdm(pool.map(mod.make_one, tasks, chunksize=8), total=len(frame), desc='EQ features'))
    else:
        features = [mod.make_one(task) for task in tqdm(tasks, total=len(frame), desc='EQ features')]
    frame['_eqtypenet_path'] = [x[0] for x in features]
    frame['ps_ratio_trace'] = [x[1] for x in features]
    frame['ps_ratio_invalid_reason'] = [x[2] for x in features]
    frame['ps_ratio_event'] = frame.groupby('event_uid')['ps_ratio_trace'].transform('median')
    frame['ps_ratio_imputation'] = np.where(frame['ps_ratio_trace'].notna(), 'none',
        np.where(frame['ps_ratio_event'].notna(), 'same_event_median', 'training_event_median'))
    frame['ps_ratio_event'] = frame['ps_ratio_event'].fillna(fallback)
    if not np.isfinite(frame['ps_ratio_event'].to_numpy(dtype=float)).all():
        raise ValueError('Invalid EQTypeNet event ratios after imputation')
    frame.to_csv(output / 'meta_evt6_test_eqtypenet.csv', index=False)
    report = {'source_manifest_sha256': sha256(data / 'meta_evt6_test.csv'),
        'training_report': str(args.train_report.resolve()), 'training_report_sha256': sha256(args.train_report),
        'training_event_ratio_median': fallback, 'samples': len(frame),
        'imputation_counts': frame['ps_ratio_imputation'].value_counts().to_dict(),
        'invalid_phase_reasons': frame['ps_ratio_invalid_reason'].value_counts().to_dict(),
        'event_ratio_note': 'Event-level median uses available records of the same external event; fallback is fixed from the DiTing training report.'}
    (output / 'preparation_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def infer(args):
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    cache = args.data.resolve(strict=True)
    source = args.source.resolve(strict=True)
    ckpt_path = args.checkpoint.resolve(strict=True)
    output = args.output.absolute()
    ensure_new(output)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0')

    if args.kind == 'eq':
        sys.path.insert(0, str(source))
        mod = load_module('eqt_external_inference', source / 'train_evt6_eqtypenet.py')
        frame = pd.read_csv(cache / 'meta_evt6_test_eqtypenet.csv', low_memory=False)
        use_ratio = not bool(state['config'].get('no_ps_ratio', False))
        dataset = mod.EQTypeNetDataset(frame, cache, use_ps_ratio=use_ratio)
        model = mod.EQTypeNetEVT6(use_ps_ratio=use_ratio)
        model.load_state_dict(state['model'], strict=True)
    else:
        sys.path.insert(0, str(source / 'evt6_capsnet_baseline'))
        mod = load_module('caps_external_inference', source / 'evt6_capsnet_baseline/train_evt6_capsnet.py')
        dataset = mod.Evt6MfccDataset(cache, 'test')
        frame = dataset.frame
        c = state['args']
        model = mod.CapsNetRes(n_classes=mod.NUM_CLASSES,
            routing_iterations=c['routing_iterations'], input_height=mod.EXPECTED_MFCC_SHAPE[1],
            input_width=mod.EXPECTED_MFCC_SHAPE[2], conv_channels=c['conv_channels'],
            primary_caps=c['primary_caps'], primary_dim=c['primary_dim'], digit_dim=c['digit_dim'],
            num_res_blocks=c['res_blocks'], dropout_p=c['dropout'])
        model.load_state_dict(state['model'], strict=True)
    del state
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    probabilities = np.empty((len(frame), 6), dtype=np.float32)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{args.kind} inference'):
            if args.kind == 'eq':
                waves, spectra, ratios, _, indices = batch
                logits = model(waves.to(device), spectra.to(device), ratios.to(device))
                probs = torch.softmax(logits, dim=1)
            else:
                features, _, indices = batch
                scores = model(features.to(device))
                probs = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            values = probs.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Non-finite model outputs')
            probabilities[indices.numpy()] = values
    predictions = frame[['trace_uid', 'event_uid']].copy()
    predictions['tgt_evt6'] = frame['_evt6'].astype(int).to_numpy()
    predictions['pred_evt6'] = probabilities.argmax(axis=1)
    for i in range(6):
        predictions[f'prob_evt6_{i}'] = probabilities[:, i]
    predictions.to_csv(output / 'predictions.csv', index=False)
    report = {'kind': args.kind, 'checkpoint': str(ckpt_path), 'checkpoint_sha256': sha256(ckpt_path),
        'samples': len(frame), 'events': int(frame['event_uid'].nunique()),
        'score_note': 'CapsNet scores are capsule activations normalized to sum to one.' if args.kind == 'caps' else 'EQTypeNet six-class softmax.'}
    (output / 'inference_metadata.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    eq = subs.add_parser('prepare-eq')
    eq.add_argument('--data', type=Path, required=True)
    eq.add_argument('--output', type=Path, required=True)
    eq.add_argument('--train-report', type=Path, required=True)
    eq.add_argument('--eq-prepare', type=Path, required=True)
    eq.add_argument('--workers', type=int, default=8)
    eq.set_defaults(func=prepare_eq)
    for kind in ('eq', 'caps'):
        infer_parser = subs.add_parser(f'infer-{kind}')
        infer_parser.add_argument('--data', type=Path, required=True)
        infer_parser.add_argument('--output', type=Path, required=True)
        infer_parser.add_argument('--source', type=Path, required=True)
        infer_parser.add_argument('--checkpoint', type=Path, required=True)
        infer_parser.add_argument('--batch-size', type=int, default=8)
        infer_parser.add_argument('--workers', type=int, default=8)
        infer_parser.set_defaults(func=infer, kind=kind)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
