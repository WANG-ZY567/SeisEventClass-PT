#!/usr/bin/env python3
"""Build a leakage-resistant DiTing 2.0 EVT6 review protocol.

Primary source policy
---------------------
* ``eq`` records come only from the natural HDF5/JSON pair.
* ``ep/ss/sp/se/ot`` records come only from the non-natural pair.
* Unknown labels and the non-``eq`` records duplicated in the natural branch
  are excluded from the primary benchmark.
* Event identity is source-qualified: ``natural_<raw event id>`` or
  ``non_<raw event id>``.  This prevents accidental cross-file event-ID
  collisions from creating train/test leakage.

All selected records pass the same strict Pg-centred crop.  No zero padding is
allowed in the primary protocol.  ``plan`` mode performs the full HDF5/QC scan
and writes planned split manifests without exporting waveform arrays.  Run it
before ``build`` mode so the final cohort can be reviewed first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


LABEL_TO_ID = {"eq": 0, "ep": 1, "ss": 2, "sp": 3, "se": 4, "ot": 5}
NON_NATURAL_LABELS = frozenset(("ep", "ss", "sp", "se", "ot"))

OUTPUT_COLUMNS = (
    "key",
    "part",
    "_src",
    "_rid",
    "_evtype",
    "_evt6",
    "_npy_path",
    "event_id_raw",
    "station_id_raw",
    "event_uid",
    "station_uid",
    "trace_uid",
    "split",
    "Pg",
    "Sg",
    "p_pick",
    "s_pick",
    "Pg_dist",
    "Sg_dist",
    "mag",
    "magtype",
    "raw_num_samples",
    "snr_db",
)

EXCLUDED_COLUMNS = (
    "source",
    "record_id",
    "event_id",
    "station_id",
    "label",
    "reason",
    "detail",
)


def _normalise_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label == "co":
        return "ss"
    return label if label in LABEL_TO_ID else "unknown"


def _split_record_id(record_id: str) -> Tuple[str, str]:
    value = str(record_id)
    if "_" not in value:
        return value, ""
    event_id, station_id = value.split("_", 1)
    return event_id, station_id


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Top-level JSON value must be an object: {path}")
    return value


def _stable_rng(seed: int, *parts: str) -> random.Random:
    payload = "|".join((str(seed),) + tuple(parts)).encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(derived)


def _sample_items(
    values: Sequence[Any], limit: int, seed: int, *parts: str
) -> List[Any]:
    ordered = sorted(values, key=str)
    if limit <= 0 or len(ordered) <= limit:
        return list(ordered)
    rng = _stable_rng(seed, *parts)
    return sorted(rng.sample(ordered, limit), key=str)


def _collect_candidates(
    natural: Mapping[str, Any],
    non_natural: Mapping[str, Any],
    eq_event_cap: int,
    eq_candidate_strategy: str,
    eq_max_traces_per_event: int,
    non_max_traces_per_event: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    natural_counts: Counter[str] = Counter()
    natural_events: Dict[str, str] = {}
    for record_id, raw_meta in natural.items():
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        label = _normalise_label(meta.get("evtype"))
        natural_counts[label] += 1
        event_id, _ = _split_record_id(str(record_id))
        previous = natural_events.setdefault(event_id, label)
        if previous != label:
            natural_events[event_id] = "mixed"

    eligible_eq_events = sorted(
        event_id for event_id, label in natural_events.items() if label == "eq"
    )
    if eq_candidate_strategy == "random":
        selected_eq_events = set(
            _sample_items(eligible_eq_events, eq_event_cap, seed, "eq_events")
        )
    elif eq_candidate_strategy == "chronological_coverage":
        selected_eq_events = set(
            _chronological_coverage_order(eligible_eq_events, seed)[:eq_event_cap]
        )
    else:
        raise ValueError(f"Unknown EQ candidate strategy: {eq_candidate_strategy}")

    grouped: Dict[Tuple[str, str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for record_id, raw_meta in natural.items():
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        label = _normalise_label(meta.get("evtype"))
        event_id, _ = _split_record_id(str(record_id))
        if label == "eq" and event_id in selected_eq_events:
            grouped[("natural", label, event_id)].append((str(record_id), meta))

    non_counts: Counter[str] = Counter()
    non_event_labels: Dict[str, str] = {}
    for record_id, raw_meta in non_natural.items():
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        label = _normalise_label(meta.get("evtype"))
        non_counts[label] += 1
        event_id, _ = _split_record_id(str(record_id))
        previous = non_event_labels.setdefault(event_id, label)
        if previous != label:
            non_event_labels[event_id] = "mixed"

    mixed_non_events = {
        event_id for event_id, label in non_event_labels.items() if label == "mixed"
    }
    for record_id, raw_meta in non_natural.items():
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        label = _normalise_label(meta.get("evtype"))
        event_id, _ = _split_record_id(str(record_id))
        if label in NON_NATURAL_LABELS and event_id not in mixed_non_events:
            grouped[("non", label, event_id)].append((str(record_id), meta))

    candidates: List[Dict[str, Any]] = []
    capped_records = Counter()
    for (source, label, event_id), records in sorted(grouped.items()):
        record_limit = (
            eq_max_traces_per_event if source == "natural" else non_max_traces_per_event
        )
        selected_records = _sample_items(
            records,
            record_limit,
            seed,
            "records",
            source,
            label,
            event_id,
        )
        capped_records[(source, label)] += len(records) - len(selected_records)
        for record_id, meta in selected_records:
            _, station_id = _split_record_id(record_id)
            candidates.append(
                {
                    "source": source,
                    "record_id": record_id,
                    "event_id": event_id,
                    "station_id": station_id,
                    "label": label,
                    "meta": meta,
                }
            )

    selection = {
        "natural_record_labels": dict(sorted(natural_counts.items())),
        "non_natural_record_labels": dict(sorted(non_counts.items())),
        "available_natural_eq_events": len(eligible_eq_events),
        "selected_natural_eq_events": len(selected_eq_events),
        "eq_candidate_strategy": eq_candidate_strategy,
        "eq_max_traces_per_event": eq_max_traces_per_event,
        "non_max_traces_per_event": non_max_traces_per_event,
        "mixed_non_natural_events_excluded": len(mixed_non_events),
        "records_removed_by_per_event_cap": {
            f"{source}:{label}": count
            for (source, label), count in sorted(capped_records.items())
        },
        "candidate_records": len(candidates),
        "source_policy_exclusions": {
            "natural_non_eq_records": int(
                sum(count for label, count in natural_counts.items() if label != "eq")
            ),
            "non_natural_unknown_or_eq_records": int(
                sum(
                    count
                    for label, count in non_counts.items()
                    if label not in NON_NATURAL_LABELS
                )
            ),
        },
    }
    return candidates, selection


def _read_three_component(handle: h5py.File, record_id: str) -> np.ndarray:
    raw = np.asarray(handle[record_id][()])
    if raw.ndim != 2:
        raise ValueError(f"not_2d:{raw.shape}")
    if raw.shape[0] == 3:
        return raw
    if raw.shape[1] == 3:
        return raw.T
    raise ValueError(f"not_three_component:{raw.shape}")


def _diagnostic_snr(waveform: np.ndarray, pg: int, window: int = 500) -> Optional[float]:
    use = min(window, pg, waveform.shape[1] - pg)
    if use <= 0:
        return None
    before = waveform[:, pg - use : pg].astype(np.float64, copy=False)
    after = waveform[:, pg : pg + use].astype(np.float64, copy=False)
    before_power = float(np.mean(np.square(before)))
    after_power = float(np.mean(np.square(after)))
    return float(10.0 * np.log10((after_power + 1e-12) / (before_power + 1e-12)))


def _safe_filename(source: str, record_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in record_id)
    return f"{source}_{safe}_0.npy"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _qc_candidates(
    candidates: Sequence[Dict[str, Any]],
    natural_h5: Path,
    non_h5: Path,
    in_samples: int,
    pre_samples: int,
    require_valid_sg: bool,
    min_post_s_samples: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter[str]]:
    eligible: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    with h5py.File(natural_h5, "r") as natural_handle, h5py.File(non_h5, "r") as non_handle:
        handles = {"natural": natural_handle, "non": non_handle}
        for index, candidate in enumerate(candidates, start=1):
            source = candidate["source"]
            record_id = candidate["record_id"]
            event_id = candidate["event_id"]
            station_id = candidate["station_id"]
            label = candidate["label"]
            meta = candidate["meta"]

            reason = ""
            detail = ""
            crop: Optional[np.ndarray] = None
            raw_num_samples: Optional[int] = None
            pg_value = _to_float(meta.get("Pg"))
            sg_value = _to_float(meta.get("Sg"))
            try:
                if pg_value is None:
                    reason = "missing_Pg"
                else:
                    waveform = _read_three_component(handles[source], record_id)
                    raw_num_samples = int(waveform.shape[1])
                    pg = int(round(pg_value))
                    crop_start = pg - pre_samples
                    crop_end = crop_start + in_samples
                    if crop_start < 0:
                        reason = "insufficient_pre_Pg"
                    elif crop_end > raw_num_samples:
                        reason = "insufficient_post_Pg"
                    else:
                        crop = waveform[:, crop_start:crop_end]
                        if crop.shape != (3, in_samples):
                            reason = "unexpected_crop_shape"
                            detail = str(crop.shape)
                        elif not np.isfinite(crop).all():
                            reason = "non_finite"
                        elif np.all(crop == 0):
                            reason = "all_zero"
                        elif np.any(np.ptp(crop.astype(np.float64, copy=False), axis=1) == 0):
                            reason = "constant_component"
                        elif require_valid_sg:
                            if sg_value is None:
                                reason = "missing_Sg"
                            else:
                                candidate_s = float(sg_value - pg_value + pre_samples)
                                if not (0 <= candidate_s < in_samples):
                                    reason = "Sg_outside_crop"
                                elif candidate_s + min_post_s_samples > in_samples:
                                    reason = "insufficient_post_Sg"
            except Exception as exc:
                reason = "hdf5_read_failed"
                detail = f"{type(exc).__name__}: {exc}"

            if reason:
                reasons[f"{source}:{label}:{reason}"] += 1
                excluded.append(
                    {
                        "source": source,
                        "record_id": record_id,
                        "event_id": event_id,
                        "station_id": station_id,
                        "label": label,
                        "reason": reason,
                        "detail": detail,
                    }
                )
                continue

            assert crop is not None and pg_value is not None and raw_num_samples is not None
            pg = int(round(pg_value))
            snr = _diagnostic_snr(waveform, pg)
            s_pick = ""
            if sg_value is not None:
                candidate_s = float(sg_value - pg_value + pre_samples)
                if 0 <= candidate_s < in_samples:
                    s_pick = candidate_s

            filename = _safe_filename(source, record_id)
            relative_path = f"waves/{filename}"

            unique_key = f"{source}_{record_id}"
            eligible.append(
                {
                    "key": unique_key,
                    "part": 0,
                    "_src": source,
                    "_rid": record_id,
                    "_evtype": label,
                    "_evt6": LABEL_TO_ID[label],
                    "_npy_path": relative_path,
                    "event_id_raw": event_id,
                    "station_id_raw": station_id,
                    "event_uid": f"{source}_{event_id}",
                    "station_uid": f"{source}_{station_id}",
                    "trace_uid": unique_key,
                    "split": "",
                    "Pg": pg_value,
                    "Sg": "" if sg_value is None else sg_value,
                    "p_pick": pre_samples,
                    "s_pick": s_pick,
                    "Pg_dist": meta.get("Pg_dist", ""),
                    "Sg_dist": meta.get("Sg_dist", ""),
                    "mag": meta.get("mag", meta.get("evmag", "")),
                    "magtype": meta.get("magtype", meta.get("mag_type", "")),
                    "raw_num_samples": raw_num_samples,
                    "snr_db": "" if snr is None else snr,
                }
            )
            if index % 1000 == 0 or index == len(candidates):
                print(
                    f"[QC] {index:,}/{len(candidates):,}; "
                    f"eligible={len(eligible):,}; excluded={len(excluded):,}",
                    flush=True,
                )

    return eligible, excluded, reasons


def _event_number_key(event_uid: str) -> Tuple[int, Any]:
    """Sort DiTing event identifiers numerically when possible."""

    raw = str(event_uid).rsplit("_", 1)[-1]
    try:
        return (0, int(raw))
    except ValueError:
        return (1, raw)


def _chronological_coverage_order(
    event_uids: Sequence[str], seed: int, num_bins: int = 100
) -> List[str]:
    """Return a deterministic order that repeatedly covers the full ID range.

    Events are divided into chronological bins.  A seeded shuffle is applied
    only within each bin, after which events are consumed round-robin across
    bins.  Prefixes of the returned order therefore span the whole catalogue
    instead of concentrating on the earliest or richest events.
    """

    ordered = sorted(event_uids, key=_event_number_key)
    if not ordered:
        return []
    bin_count = min(max(1, int(num_bins)), len(ordered))
    bins: List[List[str]] = []
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        values = ordered[start:end]
        _stable_rng(seed, "eq_chronological_bin", str(index)).shuffle(values)
        bins.append(values)

    result: List[str] = []
    depth = 0
    while True:
        added = False
        for values in bins:
            if depth < len(values):
                result.append(values[depth])
                added = True
        if not added:
            break
        depth += 1
    return result


def _select_eq_events_after_qc(
    rows: List[Dict[str, Any]],
    final_event_count: int,
    target_record_count: int,
    strategy: str,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Down-select QC-valid EQ events while retaining their real station records.

    ``match_non_natural`` matches the empirical distribution of valid records
    per event in the five non-natural classes.  It avoids selecting only the
    largest/highest-quality earthquake events while making better use of the
    available station records than an unconditioned random event sample.
    """

    eq_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    non_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = eq_by_event if row["_evtype"] == "eq" else non_by_event
        target[str(row["event_uid"])].append(row)

    available_count = len(eq_by_event)
    available_records = sum(len(value) for value in eq_by_event.values())
    keep_all_by_events = final_event_count <= 0 or final_event_count >= available_count
    keep_all_by_records = target_record_count <= 0 or target_record_count >= available_records
    if keep_all_by_events and keep_all_by_records:
        return rows, [], {
            "strategy": "all_qc_valid_eq_events",
            "available_eq_events": available_count,
            "available_eq_records": available_records,
            "selected_eq_events": available_count,
            "selected_eq_records": available_records,
            "selected_eq_record_count_distribution": dict(
                sorted(Counter(len(value) for value in eq_by_event.values()).items())
            ),
        }

    rng = _stable_rng(seed, "post_qc_eq_selection", strategy)
    available_by_count: Dict[int, List[str]] = defaultdict(list)
    for event_uid, event_rows in eq_by_event.items():
        available_by_count[len(event_rows)].append(event_uid)
    for event_uids in available_by_count.values():
        rng.shuffle(event_uids)

    desired_distribution: Counter[int] = Counter()
    if target_record_count > 0:
        if strategy != "chronological_coverage":
            raise ValueError(
                "--eq_target_records currently requires "
                "--eq_selection_strategy chronological_coverage"
            )
        coverage_order = _chronological_coverage_order(list(eq_by_event), seed)
        selected_events: set[str] = set()
        selected_records = 0
        deferred: List[str] = []
        for event_uid in coverage_order:
            size = len(eq_by_event[event_uid])
            if selected_records + size <= target_record_count:
                selected_events.add(event_uid)
                selected_records += size
            else:
                deferred.append(event_uid)

        # If an exact sum is unavailable, add the single event producing the
        # closest total.  The per-event cap bounds the maximum overshoot.
        if selected_records < target_record_count and deferred:
            best = min(
                deferred,
                key=lambda event_uid: (
                    abs(selected_records + len(eq_by_event[event_uid]) - target_record_count),
                    _event_number_key(event_uid),
                ),
            )
            if abs(selected_records + len(eq_by_event[best]) - target_record_count) < abs(
                selected_records - target_record_count
            ):
                selected_events.add(best)
    elif strategy == "random":
        selected_events = set(rng.sample(sorted(eq_by_event), final_event_count))
    elif strategy == "most_records":
        ranked = sorted(
            eq_by_event,
            key=lambda event_uid: (-len(eq_by_event[event_uid]), event_uid),
        )
        selected_events = set(ranked[:final_event_count])
    elif strategy == "match_non_natural":
        non_counts = sorted(len(event_rows) for event_rows in non_by_event.values())
        if not non_counts:
            raise ValueError("Cannot match non-natural trace distribution: no non-natural events")

        # Deterministic empirical quantiles of the non-natural trace-count profile.
        desired_counts = [
            non_counts[min(len(non_counts) - 1, int((index + 0.5) * len(non_counts) / final_event_count))]
            for index in range(final_event_count)
        ]
        rng.shuffle(desired_counts)
        desired_distribution = Counter(desired_counts)
        selected_events: set[str] = set()
        for desired in desired_counts:
            available_levels = [
                level for level, event_uids in available_by_count.items() if event_uids
            ]
            if not available_levels:
                raise RuntimeError("Ran out of QC-valid EQ events during profile matching")
            chosen_level = min(available_levels, key=lambda level: (abs(level - desired), level))
            selected_events.add(available_by_count[chosen_level].pop())
    elif strategy == "chronological_coverage":
        coverage_order = _chronological_coverage_order(list(eq_by_event), seed)
        selected_events = set(coverage_order[:final_event_count])
    else:
        raise ValueError(f"Unknown EQ selection strategy: {strategy}")

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for row in rows:
        if row["_evtype"] != "eq" or row["event_uid"] in selected_events:
            kept.append(row)
        else:
            removed.append(
                {
                    "source": row["_src"],
                    "record_id": row["_rid"],
                    "event_id": row["event_id_raw"],
                    "station_id": row["station_id_raw"],
                    "label": "eq",
                    "reason": "post_qc_eq_event_not_selected",
                    "detail": strategy,
                }
            )

    selected_distribution = Counter(
        len(eq_by_event[event_uid]) for event_uid in selected_events
    )
    return kept, removed, {
        "strategy": strategy,
        "available_eq_events": available_count,
        "available_eq_records": available_records,
        "requested_eq_events": final_event_count or None,
        "requested_eq_records": target_record_count or None,
        "selected_eq_events": len(selected_events),
        "desired_non_natural_record_count_distribution": dict(
            sorted(desired_distribution.items())
        ),
        "selected_eq_record_count_distribution": dict(
            sorted(selected_distribution.items())
        ),
        "selected_eq_records": int(
            sum(count * events for count, events in selected_distribution.items())
        ),
        "mean_records_per_selected_eq_event": float(
            sum(count * events for count, events in selected_distribution.items())
            / len(selected_events)
        ),
    }


def _export_waveforms(
    rows: Sequence[Mapping[str, Any]],
    natural_h5: Path,
    non_h5: Path,
    waves_dir: Path,
    in_samples: int,
    pre_samples: int,
) -> None:
    waves_dir.mkdir(parents=True, exist_ok=True)
    existing = next(waves_dir.iterdir(), None)
    if existing is not None:
        raise FileExistsError(
            f"Refusing to build into non-empty waveform directory: {waves_dir}"
        )

    with h5py.File(natural_h5, "r") as natural_handle, h5py.File(non_h5, "r") as non_handle:
        handles = {"natural": natural_handle, "non": non_handle}
        for index, row in enumerate(rows, start=1):
            waveform = _read_three_component(handles[str(row["_src"])], str(row["_rid"]))
            pg_value = _to_float(row["Pg"])
            if pg_value is None:
                raise RuntimeError(f"Pg disappeared during export: {row['trace_uid']}")
            pg = int(round(pg_value))
            crop_start = pg - pre_samples
            crop = waveform[:, crop_start : crop_start + in_samples]
            if crop.shape != (3, in_samples) or not np.isfinite(crop).all():
                raise RuntimeError(f"QC/export mismatch: {row['trace_uid']}: {crop.shape}")
            output_path = waves_dir / Path(str(row["_npy_path"])).name
            _atomic_save_npy(output_path, crop.astype(np.float32, copy=False))
            if index % 1000 == 0 or index == len(rows):
                print(f"[EXPORT] {index:,}/{len(rows):,}", flush=True)


def _assign_event_splits(
    rows: List[Dict[str, Any]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    strategy: str,
) -> Dict[str, str]:
    event_label: Dict[str, str] = {}
    events_by_label: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        event_uid = row["event_uid"]
        label = row["_evtype"]
        previous = event_label.setdefault(event_uid, label)
        if previous != label:
            raise ValueError(f"Mixed-label event after QC: {event_uid}: {previous}/{label}")
        events_by_label[label].add(event_uid)

    split_by_event: Dict[str, str] = {}
    test_fraction = 1.0 - train_fraction - val_fraction
    for label in sorted(events_by_label):
        events = sorted(events_by_label[label], key=_event_number_key)
        if strategy == "random":
            rng = _stable_rng(seed, "split", label)
            rng.shuffle(events)
        elif strategy != "event_number":
            raise ValueError(f"Unknown split strategy: {strategy}")
        count = len(events)
        n_val = int(round(count * val_fraction))
        n_test = int(round(count * test_fraction))
        if count >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        if n_val + n_test >= count:
            n_test = max(1, count - n_val - 1)
        n_train = count - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Not enough events to split class {label}: {count}")

        for event_uid in events[:n_train]:
            split_by_event[event_uid] = "train"
        for event_uid in events[n_train : n_train + n_val]:
            split_by_event[event_uid] = "val"
        for event_uid in events[n_train + n_val :]:
            split_by_event[event_uid] = "test"

    for row in rows:
        row["split"] = split_by_event[row["event_uid"]]
    return split_by_event


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        record_counts = Counter(str(row["_evtype"]) for row in subset)
        events_by_label: Dict[str, set[str]] = defaultdict(set)
        for row in subset:
            events_by_label[str(row["_evtype"])].add(str(row["event_uid"]))
        output[split] = {
            "num_records": len(subset),
            "num_events": len({str(row["event_uid"]) for row in subset}),
            "record_counts_by_class": dict(sorted(record_counts.items())),
            "event_counts_by_class": {
                label: len(values) for label, values in sorted(events_by_label.items())
            },
        }
    return output


def _split_event_number_ranges(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        try:
            event_number = int(str(row["event_id_raw"]))
        except (TypeError, ValueError):
            continue
        grouped[(str(row["_evtype"]), str(row["split"]))].add(event_number)

    output: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (label, split), values in sorted(grouped.items()):
        output[label][split] = {
            "num_events": len(values),
            "min": min(values),
            "max": max(values),
        }
    return dict(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reviewer-compliant DiTing 2.0 EVT6 event split"
    )
    parser.add_argument("--natural_json", type=Path, required=True)
    parser.add_argument("--natural_h5", type=Path, required=True)
    parser.add_argument("--non_json", type=Path, required=True)
    parser.add_argument("--non_h5", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "build"), default="plan")
    parser.add_argument(
        "--eq_event_cap",
        type=int,
        default=1000,
        help="EQ candidate-event pool scanned before waveform QC",
    )
    parser.add_argument(
        "--eq_candidate_strategy",
        choices=("random", "chronological_coverage"),
        default="chronological_coverage",
        help="how the pre-QC EQ event pool is selected from the catalogue",
    )
    parser.add_argument(
        "--eq_final_event_count",
        type=int,
        default=0,
        help="post-QC EQ event count; 0 keeps all QC-valid candidate events",
    )
    parser.add_argument(
        "--eq_target_records",
        type=int,
        default=0,
        help=(
            "post-QC EQ waveform target; selects whole events and allows a small "
            "bounded deviation when an exact total is unavailable"
        ),
    )
    parser.add_argument(
        "--eq_selection_strategy",
        choices=("random", "most_records", "match_non_natural", "chronological_coverage"),
        default="match_non_natural",
        help="post-QC strategy for an EQ event-count or waveform-count target",
    )
    parser.add_argument(
        "--max_traces_per_event",
        type=int,
        default=10,
        help="legacy shared per-event cap; overridden by source-specific options",
    )
    parser.add_argument(
        "--eq_max_traces_per_event",
        type=int,
        default=None,
        help="EQ per-event cap; 0 means unlimited (default: shared cap)",
    )
    parser.add_argument(
        "--non_max_traces_per_event",
        type=int,
        default=None,
        help="non-natural per-event cap; 0 means unlimited (default: shared cap)",
    )
    parser.add_argument("--in_samples", type=int, default=8192)
    parser.add_argument("--pre_samples", type=int, default=2000)
    parser.add_argument(
        "--require_valid_sg",
        action="store_true",
        help="Require Sg after Pg, inside the crop, with the requested post-S duration.",
    )
    parser.add_argument(
        "--min_post_s_samples",
        type=int,
        default=2000,
        help="Minimum samples retained after Sg when --require_valid_sg is used.",
    )
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument(
        "--split_strategy",
        choices=("random", "event_number"),
        default="random",
        help="event_number performs a class-stratified catalogue-order split",
    )
    parser.add_argument("--seed", type=int, default=20261105)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in ("natural_json", "natural_h5", "non_json", "non_h5"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if args.eq_event_cap <= 0:
        raise ValueError("--eq_event_cap must be positive")
    if args.eq_final_event_count < 0:
        raise ValueError("--eq_final_event_count must be non-negative")
    if args.eq_target_records < 0:
        raise ValueError("--eq_target_records must be non-negative")
    if args.eq_final_event_count > 0 and args.eq_target_records > 0:
        raise ValueError(
            "Use only one of --eq_final_event_count and --eq_target_records"
        )
    if args.eq_final_event_count > args.eq_event_cap:
        raise ValueError("eq_final_event_count cannot exceed eq_event_cap")
    if args.max_traces_per_event <= 0:
        raise ValueError("--max_traces_per_event must be positive")
    eq_trace_cap = (
        args.max_traces_per_event
        if args.eq_max_traces_per_event is None
        else args.eq_max_traces_per_event
    )
    non_trace_cap = (
        args.max_traces_per_event
        if args.non_max_traces_per_event is None
        else args.non_max_traces_per_event
    )
    if eq_trace_cap < 0 or non_trace_cap < 0:
        raise ValueError("source-specific per-event caps must be non-negative")
    if not (0 < args.train_fraction < 1 and 0 < args.val_fraction < 1):
        raise ValueError("train/val fractions must be in (0,1)")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    if args.pre_samples < 0 or args.pre_samples >= args.in_samples:
        raise ValueError("pre_samples must be in [0, in_samples)")
    if args.min_post_s_samples < 0 or args.min_post_s_samples >= args.in_samples:
        raise ValueError("--min_post_s_samples must be in [0, in_samples)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LOAD] natural metadata: {args.natural_json}", flush=True)
    natural = _load_json(args.natural_json)
    print(f"[LOAD] non-natural metadata: {args.non_json}", flush=True)
    non_natural = _load_json(args.non_json)

    candidates, selection = _collect_candidates(
        natural=natural,
        non_natural=non_natural,
        eq_event_cap=args.eq_event_cap,
        eq_candidate_strategy=args.eq_candidate_strategy,
        eq_max_traces_per_event=eq_trace_cap,
        non_max_traces_per_event=non_trace_cap,
        seed=args.seed,
    )
    del natural
    del non_natural
    print(f"[SELECT] candidate records: {len(candidates):,}", flush=True)

    eligible, excluded, exclusion_reasons = _qc_candidates(
        candidates=candidates,
        natural_h5=args.natural_h5,
        non_h5=args.non_h5,
        in_samples=args.in_samples,
        pre_samples=args.pre_samples,
        require_valid_sg=args.require_valid_sg,
        min_post_s_samples=args.min_post_s_samples,
    )
    if not eligible:
        raise RuntimeError("No eligible waveform records remain after QC")
    qc_eligible_before_eq_selection = len(eligible)
    qc_excluded_count = len(excluded)

    eligible, post_qc_removed, eq_post_qc_selection = _select_eq_events_after_qc(
        eligible,
        final_event_count=args.eq_final_event_count,
        target_record_count=args.eq_target_records,
        strategy=args.eq_selection_strategy,
        seed=args.seed,
    )
    excluded.extend(post_qc_removed)
    if post_qc_removed:
        exclusion_reasons["natural:eq:post_qc_eq_event_not_selected"] += len(
            post_qc_removed
        )
    if not eligible:
        raise RuntimeError("No records remain after post-QC EQ selection")

    split_by_event = _assign_event_splits(
        eligible,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        strategy=args.split_strategy,
    )

    split_events = {
        split: {event for event, assigned in split_by_event.items() if assigned == split}
        for split in ("train", "val", "test")
    }
    if split_events["train"] & split_events["val"]:
        raise AssertionError("train/val event leakage")
    if split_events["train"] & split_events["test"]:
        raise AssertionError("train/test event leakage")
    if split_events["val"] & split_events["test"]:
        raise AssertionError("val/test event leakage")

    if args.mode == "build":
        _export_waveforms(
            rows=eligible,
            natural_h5=args.natural_h5,
            non_h5=args.non_h5,
            waves_dir=args.out_dir / "waves",
            in_samples=args.in_samples,
            pre_samples=args.pre_samples,
        )

    prefix = "" if args.mode == "build" else "planned_"
    all_path = args.out_dir / f"{prefix}meta_evt6.csv"
    _write_csv(all_path, eligible, OUTPUT_COLUMNS)
    manifest_paths = [all_path]
    for split in ("train", "val", "test"):
        split_path = args.out_dir / f"{prefix}meta_evt6_{split}.csv"
        _write_csv(
            split_path,
            (row for row in eligible if row["split"] == split),
            OUTPUT_COLUMNS,
        )
        manifest_paths.append(split_path)

    excluded_path = args.out_dir / f"{prefix}excluded_records.csv"
    _write_csv(excluded_path, excluded, EXCLUDED_COLUMNS)
    manifest_paths.append(excluded_path)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source_policy": {
            "eq": "natural JSON/HDF5 only",
            "ep_ss_sp_se_ot": "non-natural JSON/HDF5 only",
            "unknown_labels": "excluded",
            "overlapping_non_natural_records_in_natural_file": "excluded from primary benchmark",
            "event_uid": "natural_<raw event id> or non_<raw event id>",
        },
        "preprocessing": {
            "sampling_rate_hz": 50,
            "in_samples": args.in_samples,
            "pre_Pg_samples": args.pre_samples,
            "post_Pg_samples": args.in_samples - args.pre_samples,
            "window_seconds": args.in_samples / 50.0,
            "require_valid_Sg": bool(args.require_valid_sg),
            "minimum_post_Sg_samples": (
                args.min_post_s_samples if args.require_valid_sg else None
            ),
            "zero_padding": False,
            "additional_filtering": False,
            "resampling": False,
            "saved_dtype": "float32" if args.mode == "build" else None,
            "runtime_normalization": "per-component demean and standard deviation in training/preprocess.py",
        },
        "selection": {
            **selection,
            "post_qc_eq_selection": eq_post_qc_selection,
        },
        "qc": {
            "eligible_records_before_post_qc_eq_selection": qc_eligible_before_eq_selection,
            "qc_excluded_candidate_records": qc_excluded_count,
            "final_eligible_records": len(eligible),
            "post_qc_eq_records_not_selected": len(post_qc_removed),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        },
        "split": {
            "seed": args.seed,
            "strategy": args.split_strategy,
            "ordering_field": (
                "numeric event_id_raw (catalogue-order proxy)"
                if args.split_strategy == "event_number"
                else None
            ),
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
            "summary": _split_summary(eligible),
            "event_number_ranges_by_class": _split_event_number_ranges(eligible),
            "event_overlap_counts": {
                "train_val": len(split_events["train"] & split_events["val"]),
                "train_test": len(split_events["train"] & split_events["test"]),
                "val_test": len(split_events["val"] & split_events["test"]),
            },
        },
        "manifest_sha256": {path.name: _sha256(path) for path in manifest_paths},
    }
    report_path = args.out_dir / f"{prefix}protocol_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report["qc"], ensure_ascii=False, indent=2))
    print(json.dumps(report["split"]["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Protocol report: {report_path}")
    print(f"[OK] Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
