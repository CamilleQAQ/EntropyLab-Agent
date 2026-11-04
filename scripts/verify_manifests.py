#!/usr/bin/env python
"""Independent verification of the frozen Paderborn Phase 1 manifests.

This script does not rebuild the split. It verifies the written contracts and
hashes, so it can be run before every later phase and in CI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


WORKSPACE = Path(r"E:\Research\EntropyLab Agent")
ROOT = WORKSPACE / "renovation"
MANIFESTS = ROOT / "manifests" / "paderborn"
PARTITIONS = [
    "initial_train",
    "initial_validation",
    "batch_01",
    "batch_02",
    "final_holdout",
]
EXPECTED_RECORDS = {
    "initial_train": 640,
    "initial_validation": 320,
    "batch_01": 320,
    "batch_02": 320,
    "final_holdout": 480,
}
EXPECTED_CLASSES = {
    "initial_train": {"HEALTHY": 160, "OUTER_RACE": 240, "INNER_RACE": 240},
    "initial_validation": {"HEALTHY": 80, "OUTER_RACE": 160, "INNER_RACE": 80},
    "batch_01": {"HEALTHY": 80, "OUTER_RACE": 160, "INNER_RACE": 80},
    "batch_02": {"HEALTHY": 80, "OUTER_RACE": 80, "INNER_RACE": 160},
    "final_holdout": {"HEALTHY": 80, "OUTER_RACE": 240, "INNER_RACE": 160},
}
COLUMNS = [
    "record_id",
    "path",
    "label",
    "label_id",
    "bearing_id",
    "condition_id",
    "recording_number",
    "group_id",
    "partition",
    "source_archive",
    "archive_sha256",
    "raw_member",
    "raw_sha256",
    "raw_bytes",
    "acquisition_datetime_local_unzoned",
    "timezone",
    "sampling_rate_hz",
    "signal_name",
    "vibration_samples",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    summary = json.loads((MANIFESTS / "split_summary.json").read_text(encoding="utf-8"))
    data_root = Path(summary["data_root"])

    expected_hashes = summary.get("manifest_sha256") or {}
    for name in [f"{part}.csv" for part in PARTITIONS] + ["all_records.csv"]:
        path = MANIFESTS / name
        if not path.exists():
            errors.append(f"missing manifest: {name}")
            continue
        actual = sha256(path)
        expected = expected_hashes.get(name)
        if expected != actual:
            errors.append(f"{name}: SHA-256 {actual} != frozen {expected}")

    frames = {}
    for part in PARTITIONS:
        path = MANIFESTS / f"{part}.csv"
        if not path.exists():
            errors.append(f"missing partition manifest: {part}")
            continue
        frame = pd.read_csv(path)
        frames[part] = frame
        if list(frame.columns) != COLUMNS:
            errors.append(f"{part}: columns differ from contract")
        if len(frame) != EXPECTED_RECORDS[part]:
            errors.append(f"{part}: {len(frame)} records != {EXPECTED_RECORDS[part]}")
        class_counts = Counter(frame["label"])
        if dict(class_counts) != EXPECTED_CLASSES[part]:
            errors.append(f"{part}: class counts {dict(class_counts)} != {EXPECTED_CLASSES[part]}")
        if frame["partition"].nunique() != 1 or frame["partition"].iloc[0] != part:
            errors.append(f"{part}: partition column is inconsistent")
        if frame["record_id"].duplicated().any():
            errors.append(f"{part}: duplicate record_id")
        if frame["raw_sha256"].isna().any() or (frame["raw_sha256"].astype(str).str.len() != 64).any():
            errors.append(f"{part}: missing or invalid raw_sha256")
        if not set(frame["condition_id"]) == {
            "N09_M07_F10",
            "N15_M01_F10",
            "N15_M07_F04",
            "N15_M07_F10",
        }:
            errors.append(f"{part}: conditions are incomplete")
        if not set(frame["sampling_rate_hz"]) == {64000}:
            errors.append(f"{part}: sampling rate is not 64000")
        if not set(frame["signal_name"]) == {"vibration_1"}:
            errors.append(f"{part}: signal_name is not vibration_1")

        for row in frame.itertuples(index=False):
            path_value = data_root / str(row.path)
            if not path_value.exists():
                errors.append(f"{part}: missing signal file {row.path}")
                break

    # Group isolation and class coverage.
    bearing_to_partition: dict[str, str] = {}
    for part, frame in frames.items():
        for bearing in set(frame["bearing_id"]):
            if bearing in bearing_to_partition:
                errors.append(f"{bearing}: appears in {bearing_to_partition[bearing]} and {part}")
            bearing_to_partition[bearing] = part
    if len(bearing_to_partition) != 26:
        errors.append(f"expected 26 bearings, found {len(bearing_to_partition)}")

    # All records must be exactly the union of the partition manifests.
    all_records = pd.read_csv(MANIFESTS / "all_records.csv")
    if len(all_records) != 2080:
        errors.append(f"all_records.csv: {len(all_records)} != 2080")
    if set(all_records["record_id"]) != set().union(*(set(f["record_id"]) for f in frames.values())):
        errors.append("all_records.csv is not the union of the partition manifests")
    if all_records["record_id"].duplicated().any():
        errors.append("all_records.csv contains duplicate record_id values")

    # Cross-partition content duplicates are leakage.
    hash_partitions: dict[str, set[str]] = {}
    for part, frame in frames.items():
        for digest in frame["raw_sha256"].astype(str):
            hash_partitions.setdefault(digest, set()).add(part)
    leaked = {digest: sorted(parts) for digest, parts in hash_partitions.items() if len(parts) > 1}
    if leaked:
        errors.append(f"{len(leaked)} content hashes cross partitions")

    # Assignments file must agree with the manifests.
    assignments = pd.read_csv(MANIFESTS / "assignments.csv")
    if len(assignments) != 26:
        errors.append(f"assignments.csv: {len(assignments)} bearings != 26")
    assign_map = dict(zip(assignments["bearing_id"], assignments["partition"]))
    if assign_map != bearing_to_partition:
        errors.append("assignments.csv disagrees with the partition manifests")

    print("Paderborn Phase 1 manifest verification")
    print(f"data root: {data_root}")
    print(f"partitions: { {p: len(frames[p]) for p in frames} }")
    print(f"bearings: {len(bearing_to_partition)}")
    print(f"errors: {len(errors)}")
    for error in errors[:30]:
        print("ERROR:", error)
    print("STATUS:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
