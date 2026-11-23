#!/usr/bin/env python3
"""Read-only checks before changing the EVT6 training recipe."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ("eq", "ep", "ss", "sp", "se", "ot")
PROB_COLUMNS = [f"prob_evt6_{i}" for i in range(len(CLASSES))]


def audit_manifest(path: Path) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"_evt6", "event_uid", "Pg", "Sg", "s_pick", "trace_uid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    labels = pd.to_numeric(frame["_evt6"], errors="coerce")
    if labels.isna().any() or not labels.between(0, 5).all():
        raise ValueError(f"{path}: invalid _evt6 values")
    if frame["trace_uid"].duplicated().any():
        raise ValueError(f"{path}: duplicate trace_uid values")
    sources = (frame["_src"] if "_src" in frame else
               frame["trace_uid"].astype(str).str.split("_", n=1).str[0])
    source_by_class = {}
    for class_id, class_name in enumerate(CLASSES):
        counts = sources[labels == class_id].value_counts()
        source_by_class[class_name] = {str(k): int(v) for k, v in counts.items()}
    pg = pd.to_numeric(frame["Pg"], errors="coerce")
    sg = pd.to_numeric(frame["Sg"], errors="coerce")
    s_pick = pd.to_numeric(frame["s_pick"], errors="coerce")
    valid = pg.notna() & sg.notna()
    bad_order = valid & sg.le(pg)
    counts = {
        "rows": len(frame),
        "events": int(frame["event_uid"].nunique()),
        "class_records": {name: int((labels == i).sum()) for i, name in enumerate(CLASSES)},
        "source_by_class": source_by_class,
        "missing_pg": int(pg.isna().sum()),
        "missing_sg": int(sg.isna().sum()),
        "sg_not_after_pg": int(bad_order.sum()),
        "s_pick_before_p_pick": int((s_pick < pd.to_numeric(frame.get("p_pick", 500), errors="coerce")).sum()),
    }
    return counts, frame


def audit_predictions(path: Path, manifest: pd.DataFrame) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    required = {"trace_uid", "event_uid", "tgt_evt6", "pred_evt6", *PROB_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; rerun validation with --save-test-probs true")
    if frame["trace_uid"].duplicated().any():
        raise ValueError(f"{path}: duplicate trace_uid values")
    target = frame["tgt_evt6"].to_numpy(dtype=int)
    pred = frame["pred_evt6"].to_numpy(dtype=int)
    probs = frame[PROB_COLUMNS].to_numpy(dtype=float)
    if (target < 0).any() or (target >= len(CLASSES)).any():
        raise ValueError(f"{path}: target outside EVT6 class range")
    if not np.isfinite(probs).all():
        raise ValueError(f"{path}: non-finite probabilities")
    true_prob = probs[np.arange(len(frame)), target]
    wrong = target != pred
    thresholds = (0.0, 1e-8, 1e-6, 1e-4)
    result = {
        "rows": len(frame),
        "events": int(frame["event_uid"].nunique()),
        "accuracy": float((~wrong).mean()),
        "wrong_rows": int(wrong.sum()),
        "true_probability_below": {
            str(t): {"all": int((true_prob <= t).sum()),
                     "wrong": int(((true_prob <= t) & wrong).sum())}
            for t in thresholds
        },
        "wrong_by_true_class": {name: int((wrong & (target == i)).sum())
                                for i, name in enumerate(CLASSES)},
    }
    errors = Counter((CLASSES[int(t)], CLASSES[int(p)])
                     for t, p in zip(target[wrong], pred[wrong]))
    result["largest_confusions"] = [
        {"true": a, "predicted": b, "count": n}
        for (a, b), n in errors.most_common(10)
    ]
    manifest_ids = set(manifest["trace_uid"].astype(str))
    prediction_ids = set(frame["trace_uid"].astype(str))
    result["manifest_match"] = {
        "prediction_not_in_manifest": len(prediction_ids - manifest_ids),
        "manifest_not_in_prediction": len(manifest_ids - prediction_ids),
    }
    event_errors = defaultdict(list)
    for uid, bad in zip(frame["event_uid"].astype(str), wrong):
        event_errors[uid].append(bool(bad))
    result["events_with_any_error"] = sum(any(v) for v in event_errors.values())
    result["events_all_wrong"] = sum(all(v) for v in event_errors.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    report = {"data_dir": str(args.data_dir), "manifests": {}}
    manifests = {}
    all_events = {}
    for split in ("train", "val", "test"):
        path = args.data_dir / f"meta_evt6_{split}.csv"
        summary, frame = audit_manifest(path)
        report["manifests"][split] = summary
        manifests[split] = frame
        all_events[split] = set(frame["event_uid"].astype(str))
    report["event_overlap"] = {
        f"{a}_{b}": len(all_events[a] & all_events[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if args.val_csv:
        report["validation_predictions"] = audit_predictions(args.val_csv, manifests["val"])
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
