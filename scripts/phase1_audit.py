#!/usr/bin/env python
"""Phase 1 audit for the Paderborn three-class bearing track.

The script is deliberately independent of the old record-level split.  It
rebuilds the closed set from the official archive inventory, verifies every
selected MAT file, assigns complete bearings to the frozen partitions, and
writes manifests plus a provenance report.

Run from the renovation workspace:

    python scripts/phase1_audit.py --hash --deep
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


WORKSPACE = Path(r"E:\Research\EntropyLab Agent")
RENOVATION = WORKSPACE / "renovation"
DATA_ROOT = RENOVATION / "data" / "paderborn"
RAW = DATA_ROOT / "raw"
OUT = RENOVATION / "manifests" / "paderborn"

SOURCE_VERIFICATION = WORKSPACE / "Paderborn-source-verification-20261102"
ARCHIVES = SOURCE_VERIFICATION / "archives"
BEARINGS_CSV = WORKSPACE / "entropy-agent" / "docs" / "paderborn_full_acceptance_20261102" / "bearings.csv"
CONDITIONS_CSV = WORKSPACE / "entropy-agent" / "docs" / "paderborn_full_acceptance_20261102" / "conditions.csv"
OLD_RECORDS_CSV = (
    WORKSPACE
    / "entropy-agent"
    / "results"
    / "paderborn-e2e-20261102T133445Z-9eee8c"
    / "data"
    / "records.csv"
)
FULL_RECORDS_JSON = (
    WORKSPACE
    / "_archive"
    / "entropy-agent-20261104"
    / "docs"
    / "paderborn_full_acceptance_20261102"
    / "records.json"
)

LABELS = {"HEALTHY": 0, "OUTER_RACE": 1, "INNER_RACE": 2}
CLASS_NAMES = {value: key for key, value in LABELS.items()}
CONDITIONS = [
    "N09_M07_F10",
    "N15_M01_F10",
    "N15_M07_F04",
    "N15_M07_F10",
]

# Pre-registered at bearing level.  No model score, validation score or test
# score was used to choose these lists.  The assignment balances damage origin
# within each available class.
SPLIT = {
    "initial_train": ["K001", "K003", "KA01", "KA04", "KA06", "KI01", "KI03", "KI16"],
    "initial_validation": ["K005", "KA05", "KA15", "KI05"],
    "batch_01": ["K002", "KA03", "KA16", "KI17"],
    "batch_02": ["K004", "KA07", "KI07", "KI18"],
    "final_holdout": ["K006", "KA09", "KA22", "KA30", "KI08", "KI21"],
}

EXCLUDED = {
    "KB23": "combined inner+outer damage; not a pure class",
    "KB24": "combined inner+outer damage; not a pure class",
    "KB27": "combined inner+outer damage; not a pure class",
    "KI04": "factsheet records combined inner+outer damage",
    "KI14": "factsheet records combined inner+outer damage",
    "KA08": "outer-race label ambiguity; one MAT record is unreadable",
}

UNKNOWN_FIELDS = [
    "physical_machine_id",
    "machine_serial_number",
    "lifetime_test_id",
    "lifetime_test_batch_id",
    "datetime_timezone",
    "signal_unit",
    "shifted_or_partial_cross_record_overlap",
]

PARTITION_ORDER = [
    "initial_train",
    "initial_validation",
    "batch_01",
    "batch_02",
    "final_holdout",
]

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


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def mat_path(raw_member: str) -> Path:
    return RAW.joinpath(*raw_member.replace("\\", "/").split("/"))


def deep_check(path: Path, expected_samples: int) -> dict:
    value = sio.loadmat(path)
    keys = [key for key in value if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"expected one top-level MAT key, found {keys}")
    struct = value[keys[0]]
    if struct.shape != (1, 1) or struct.dtype.names is None:
        raise ValueError("top-level MAT value is not a 1x1 struct")
    if "Y" not in struct.dtype.names:
        raise ValueError("MAT struct has no Y field")
    channels = struct[0, 0]["Y"]
    if channels.dtype.names is None:
        raise ValueError("Y is not a struct array")
    names = [str(channels[0, index]["Name"][0]) for index in range(channels.shape[1])]
    if "vibration_1" not in names:
        raise ValueError(f"vibration_1 missing; channels={names}")
    channel = channels[0, names.index("vibration_1")]
    data = np.asarray(channel["Data"]).squeeze()
    if data.ndim != 1:
        raise ValueError(f"vibration_1 is not one-dimensional: {data.shape}")
    if len(data) != expected_samples:
        raise ValueError(f"vibration_samples {expected_samples} != actual {len(data)}")
    if not np.isfinite(data).all():
        raise ValueError("vibration_1 contains NaN or infinity")
    x_index = np.asarray(channel["XIndex"]).squeeze()
    if x_index.size and not np.isfinite(x_index).all():
        raise ValueError("vibration_1 XIndex contains NaN or infinity")
    if x_index.size > 1 and not np.all(np.diff(x_index.astype(float)) > 0):
        raise ValueError("vibration_1 XIndex is not strictly increasing")
    return {
        "mat_key": str(keys[0]),
        "channels": names,
        "samples": int(len(data)),
        "finite": True,
        "xindex_monotonic": bool(x_index.size <= 1 or np.all(np.diff(x_index.astype(float)) > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="hash every selected MAT file")
    parser.add_argument("--deep", action="store_true", help="load every selected MAT and verify vibration_1")
    parser.add_argument("--no-manifests", action="store_true", help="audit only; do not rewrite manifests")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    if not RAW.exists():
        errors.append(f"raw data directory is missing: {RAW}")
        print("\n".join(errors))
        return 2

    bearings = pd.read_csv(BEARINGS_CSV)
    conditions = pd.read_csv(CONDITIONS_CSV)
    full = load_json(FULL_RECORDS_JSON)
    old = pd.read_csv(OLD_RECORDS_CSV)

    selected_bearings = [bearing for group in SPLIT.values() for bearing in group]
    if len(selected_bearings) != len(set(selected_bearings)):
        errors.append("a bearing appears in more than one partition")
    if set(selected_bearings) & set(EXCLUDED):
        errors.append("an excluded bearing appears in the split")
    inventory = set(bearings["bearing_id"])
    covered = set(selected_bearings) | set(EXCLUDED)
    if covered != inventory:
        missing = sorted(inventory - covered)
        extra = sorted(covered - inventory)
        errors.append(f"bearing selection does not partition the inventory; missing={missing} extra={extra}")

    bearing_info = bearings.set_index("bearing_id").to_dict("index")
    expected_class_counts = {
        "initial_train": Counter({"HEALTHY": 2, "OUTER_RACE": 3, "INNER_RACE": 3}),
        "initial_validation": Counter({"HEALTHY": 1, "OUTER_RACE": 2, "INNER_RACE": 1}),
        "batch_01": Counter({"HEALTHY": 1, "OUTER_RACE": 2, "INNER_RACE": 1}),
        "batch_02": Counter({"HEALTHY": 1, "OUTER_RACE": 1, "INNER_RACE": 2}),
        "final_holdout": Counter({"HEALTHY": 1, "OUTER_RACE": 3, "INNER_RACE": 2}),
    }
    for partition, members in SPLIT.items():
        actual = Counter(str(bearing_info[b]["paper_summary_label"]) for b in members)
        if actual != expected_class_counts[partition]:
            errors.append(
                f"{partition} class counts {dict(actual)} != expected {dict(expected_class_counts[partition])}"
            )

    archive_hashes: dict[str, str] = {}
    for archive in sorted(ARCHIVES.glob("*.rar")):
        archive_hashes[archive.stem] = sha256_file(archive)
    if len(archive_hashes) != 32:
        errors.append(f"expected 32 archives, found {len(archive_hashes)}")

    full_by_member = {str(record["raw_member"]): record for record in full}
    selected_records = [
        record for record in full if str(record["bearing_id"]) in set(selected_bearings)
    ]
    if len(selected_records) != 2080:
        errors.append(f"expected 2080 selected records, found {len(selected_records)}")

    # Validate the closed-set labels and the official inventory link.
    for record in selected_records:
        bearing = str(record["bearing_id"])
        label = str(record["label"])
        if label not in LABELS:
            errors.append(f"{record.get('record_id')}: unsupported label {label!r}")
            continue
        if int(record["label_id"]) != LABELS[label]:
            errors.append(f"{record.get('record_id')}: label_id does not match {label}")
        if str(record["bearing_id"]) != bearing:
            errors.append(f"{record.get('record_id')}: bearing mismatch")
        if record.get("archive_sha256") and record["archive_sha256"] != archive_hashes.get(bearing):
            errors.append(f"{bearing}: archive hash mismatch for {record.get('raw_member')}")
        member_name = Path(str(record["raw_member"]).replace("\\", "/")).stem
        expected_stem = (
            f"{record['condition_id']}_{bearing}_{int(record['recording_number'])}"
        )
        if member_name != expected_stem:
            errors.append(
                f"{record.get('record_id')}: filename stem {member_name!r} != {expected_stem!r}"
            )
        if not bool(record.get("key_matches_filename", False)):
            errors.append(f"{record.get('record_id')}: MAT key does not match filename")

    inventory_status_counts = Counter(str(record.get("status")) for record in selected_records)
    inventory_approval_counts = Counter(
        str(record.get("formal_approval")) for record in selected_records
    )

    # Count expected records per bearing, condition and recording number.
    by_bearing: dict[str, list[dict]] = defaultdict(list)
    for record in selected_records:
        by_bearing[str(record["bearing_id"])].append(record)
    for bearing in selected_bearings:
        records = by_bearing.get(bearing, [])
        if len(records) != 80:
            errors.append(f"{bearing}: expected 80 records, found {len(records)}")
        condition_counts = Counter(str(record["condition_id"]) for record in records)
        if set(condition_counts) != set(CONDITIONS):
            errors.append(f"{bearing}: conditions {sorted(condition_counts)} != {CONDITIONS}")
        for condition in CONDITIONS:
            numbers = sorted(
                int(record["recording_number"])
                for record in records
                if str(record["condition_id"]) == condition
            )
            if numbers != list(range(1, 21)):
                errors.append(f"{bearing}/{condition}: recording numbers are not 1..20")

    # Verify every selected MAT by path, size and (optionally) SHA-256.
    hash_rows: list[dict] = []
    hash_by_member: dict[str, str] = {}
    missing_files: list[str] = []
    deep_failures: list[str] = []
    for index, record in enumerate(sorted(selected_records, key=lambda item: str(item["raw_member"]))):
        member = str(record["raw_member"])
        path = mat_path(member)
        expected_hash = str(record["raw_file_id"]).removeprefix("sha256:")
        if not path.exists():
            missing_files.append(member)
            continue
        if path.stat().st_size != int(record["raw_bytes"]):
            errors.append(f"{member}: size {path.stat().st_size} != {record['raw_bytes']}")
        if args.hash:
            actual_hash = sha256_file(path)
            hash_by_member[member] = actual_hash
            hash_rows.append(
                {
                    "raw_member": member,
                    "raw_sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "match": actual_hash == expected_hash,
                    "raw_bytes": path.stat().st_size,
                }
            )
            if actual_hash != expected_hash:
                errors.append(f"{member}: SHA-256 mismatch")
        if args.deep:
            try:
                deep_check(path, int(record["vibration_samples"]))
            except Exception as exc:  # noqa: BLE001 - audit must report the exact file
                deep_failures.append(f"{member}: {exc}")
        if (index + 1) % 200 == 0:
            print(f"checked {index + 1}/{len(selected_records)} selected MAT files")

    if missing_files:
        errors.append(f"{len(missing_files)} selected MAT files are missing; first={missing_files[:5]}")
    if deep_failures:
        errors.extend(deep_failures[:20])
        if len(deep_failures) > 20:
            errors.append(f"... and {len(deep_failures) - 20} more deep-check failures")

    # Build the record rows after the file audit, so a failed audit cannot
    # silently produce a "clean" manifest.
    rows: list[dict] = []
    for record in selected_records:
        bearing = str(record["bearing_id"])
        label = str(record["label"])
        partition = next(
            name for name, members in SPLIT.items() if bearing in members
        )
        member = str(record["raw_member"])
        rows.append(
            {
                "record_id": str(record["recording_id"]),
                "path": f"raw/{member}",
                "label": label,
                "label_id": LABELS[label],
                "bearing_id": bearing,
                "condition_id": str(record["condition_id"]),
                "recording_number": int(record["recording_number"]),
                "group_id": f"PU:{bearing}",
                "partition": partition,
                "source_archive": f"{bearing}.rar",
                "archive_sha256": str(record["archive_sha256"]),
                "raw_member": member,
                "raw_sha256": hash_by_member.get(member, ""),
                "raw_bytes": int(record["raw_bytes"]),
                "acquisition_datetime_local_unzoned": str(
                    record.get("acquisition_datetime_local_unzoned", "")
                ),
                "timezone": "UNKNOWN",
                "sampling_rate_hz": int(record.get("nominal_vibration_fs_hz", 64000)),
                "signal_name": "vibration_1",
                "vibration_samples": int(record["vibration_samples"]),
            }
        )
    rows.sort(
        key=lambda row: (
            PARTITION_ORDER.index(row["partition"]),
            row["bearing_id"],
            row["condition_id"],
            row["recording_number"],
        )
    )

    # Cross-partition duplicate content is leakage; same-partition duplicates
    # are reported but not automatically fatal.
    hash_to_partitions: dict[str, set[str]] = defaultdict(set)
    hash_to_records: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if not row["raw_sha256"]:
            continue
        hash_to_partitions[row["raw_sha256"]].add(row["partition"])
        hash_to_records[row["raw_sha256"]].append(row["record_id"])
    cross_partition_duplicates = {
        digest: sorted(parts)
        for digest, parts in hash_to_partitions.items()
        if len(parts) > 1
    }
    if cross_partition_duplicates:
        errors.append(
            f"{len(cross_partition_duplicates)} content hashes appear in multiple partitions"
        )

    # Compare against the old exploratory manifest without reusing its split.
    old_paths = {
        str(path).replace("\\", "/").removeprefix("raw/")
        for path in old["path"].astype(str)
    }
    new_paths = {row["raw_member"] for row in rows}
    old_missing = sorted(new_paths - old_paths)
    old_extra = sorted(old_paths - new_paths)
    if len(old_paths) != 2076:
        warnings.append(f"old exploratory manifest has {len(old_paths)} records, expected 2076")
    if old_missing or old_extra:
        warnings.append(
            f"old manifest differs: new-only={len(old_missing)} old-only={len(old_extra)}"
        )
    old_missing_details = [
        {
            "record_id": row["record_id"],
            "raw_member": row["raw_member"],
            "label": row["label"],
            "bearing_id": row["bearing_id"],
            "condition_id": row["condition_id"],
            "raw_sha256": row["raw_sha256"],
        }
        for row in rows
        if row["raw_member"] in set(old_missing)
    ]

    # A failed audit must not produce a "best effort" split.
    if errors:
        dump_json(
            OUT / "audit_errors.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "errors": errors,
                "warnings": warnings,
                "missing_files": missing_files[:50],
                "deep_failures": deep_failures[:50],
                "cross_partition_duplicates": sorted(cross_partition_duplicates)[:50],
            },
        )
        print(f"status: FAIL")
        print(f"errors: {len(errors)} warnings: {len(warnings)}")
        for error in errors[:30]:
            print("ERROR:", error)
        return 1

    # Write manifests only after the audit has no fatal errors.
    manifest_sha256: dict[str, str] = {}
    split_summary = None
    if not args.no_manifests:
        write_csv(OUT / "all_records.csv", rows, COLUMNS)
        for partition in PARTITION_ORDER:
            part_rows = [row for row in rows if row["partition"] == partition]
            write_csv(OUT / f"{partition}.csv", part_rows, COLUMNS)
        assignments = []
        for partition in PARTITION_ORDER:
            for bearing in SPLIT[partition]:
                info = bearing_info[bearing]
                records = by_bearing[bearing]
                assignments.append(
                    {
                        "bearing_id": bearing,
                        "label": info["paper_summary_label"],
                        "label_id": LABELS[str(info["paper_summary_label"])],
                        "damage_origin": info["damage_origin"],
                        "partition": partition,
                        "n_records": len(records),
                        "conditions": "|".join(
                            sorted({str(record["condition_id"]) for record in records})
                        ),
                    }
                )
        write_csv(
            OUT / "assignments.csv",
            assignments,
            [
                "bearing_id",
                "label",
                "label_id",
                "damage_origin",
                "partition",
                "n_records",
                "conditions",
            ],
        )

        for path in sorted(OUT.glob("*.csv")):
            if path.name == "assignments.csv":
                continue
            manifest_sha256[path.name] = sha256_file(path)

        split_summary = {
            "dataset_version": "paderborn-3class-26bearings-20261104",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(DATA_ROOT),
            "source_archives": str(ARCHIVES),
            "partitions": {},
            "totals": {},
            "manifest_sha256": manifest_sha256,
            "selection": {
                "selected_bearings": selected_bearings,
                "excluded_bearings": EXCLUDED,
            },
            "unknown_fields": UNKNOWN_FIELDS,
        }
        for partition in PARTITION_ORDER:
            part_rows = [row for row in rows if row["partition"] == partition]
            split_summary["partitions"][partition] = {
                "bearings": SPLIT[partition],
                "bearing_count": len(SPLIT[partition]),
                "record_count": len(part_rows),
                "class_counts": dict(Counter(row["label"] for row in part_rows)),
                "condition_counts": dict(Counter(row["condition_id"] for row in part_rows)),
                "origin_counts": dict(
                    Counter(
                        str(bearing_info[bearing]["damage_origin"])
                        for bearing in SPLIT[partition]
                    )
                ),
            }
        split_summary["totals"] = {
            "bearing_count": len(selected_bearings),
            "record_count": len(rows),
            "class_counts": dict(Counter(row["label"] for row in rows)),
            "condition_counts": dict(Counter(row["condition_id"] for row in rows)),
            "archive_count": len(archive_hashes),
            "archive_bytes": sum((ARCHIVES / f"{name}.rar").stat().st_size for name in archive_hashes),
        }
        dump_json(OUT / "split_summary.json", split_summary)

        provenance = {
            "source": {
                "name": "Paderborn Bearing Data Center",
                "url": "https://groups.uni-paderborn.de/kat/BearingDataCenter/",
                "license": "CC BY-NC 4.0 (official data page; verify before redistribution)",
                "local_archives": str(ARCHIVES),
                "archive_count": len(archive_hashes),
                "archive_sha256": archive_hashes,
            },
            "extraction": {
                "data_root": str(DATA_ROOT),
                "raw_root": str(RAW),
                "expected_records": 2080,
                "selected_records": len(rows),
                "selected_bearings": len(selected_bearings),
                "classes": LABELS,
                "conditions": CONDITIONS,
                "signal_name": "vibration_1",
                "nominal_sampling_rate_hz": 64000,
            },
            "selection_rule": {
                "closed_set": "three classes with a single supported label",
                "excluded": EXCLUDED,
                "not_used": "old record-level split; superseded by bearing-level assignment",
            },
            "audit": {
                "hash_enabled": bool(args.hash),
                "deep_signal_check": bool(args.deep),
                "inventory_status_counts": dict(inventory_status_counts),
                "inventory_formal_approval_counts": dict(inventory_approval_counts),
                "formal_approval_note": (
                    "The independent inventory marks every selected record as "
                    "formal_approval=false and status=STRUCTURE_AUDITED_NOT_"
                    "TRAINING_APPROVED. This Phase 1 result is an internal provenance "
                    "and split freeze, not a formal dataset/training approval."
                ),
                "missing_files": len(missing_files),
                "deep_failures": len(deep_failures),
                "cross_partition_duplicate_hashes": len(cross_partition_duplicates),
                "errors": errors,
                "warnings": warnings,
                "old_manifest_new_only": old_missing,
                "old_manifest_old_only": old_extra,
                "old_manifest_new_only_details": old_missing_details,
                "old_manifest_note": (
                    "The four new-only records are all KA04 and were omitted by the "
                    "old exploratory manifest without a recorded reason. They pass "
                    "the independent byte-hash and deep signal checks here. Restoring "
                    "them is a deliberate audit result, not a silent row addition."
                ),
            },
            "unknown_fields": UNKNOWN_FIELDS,
        }
        dump_json(OUT / "provenance.json", provenance)

        report = [
            "# Paderborn Phase 1 audit",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"**Status: {'PASS' if not errors else 'FAIL'}**",
            "",
            "## Counts",
            "",
            "| Partition | Bearings | Records | Classes |",
            "|---|---:|---:|---|",
        ]
        for partition in PARTITION_ORDER:
            part = split_summary["partitions"][partition]
            report.append(
                f"| `{partition}` | {part['bearing_count']} | {part['record_count']} | "
                f"{part['class_counts']} |"
            )
        report.extend(
            [
                "",
                f"Total: **{len(rows)} records / {len(selected_bearings)} bearings**.",
                "",
                "## Checks",
                "",
                f"- archives: {len(archive_hashes)}",
                f"- selected MAT missing: {len(missing_files)}",
                f"- SHA-256 mismatches: {sum(1 for row in hash_rows if not row['match'])}",
                f"- deep signal failures: {len(deep_failures)}",
                f"- cross-partition duplicate hashes: {len(cross_partition_duplicates)}",
                f"- errors: {len(errors)}",
                f"- warnings: {len(warnings)}",
                "",
                "## Inventory status",
                "",
                "The independent inventory marks every selected record as "
                "`formal_approval=false` and "
                "`status=STRUCTURE_AUDITED_NOT_TRAINING_APPROVED`. This Phase 1 "
                "result is an internal provenance/split freeze, not a formal "
                "dataset or training approval.",
                "",
                f"- inventory status: {dict(inventory_status_counts)}",
                f"- formal approval: {dict(inventory_approval_counts)}",
                "",
                "## Excluded bearings",
                "",
            ]
        )
        for bearing, reason in EXCLUDED.items():
            report.append(f"- `{bearing}`: {reason}")
        report.extend(
            [
                "",
                "## Unknown fields",
                "",
                ", ".join(f"`{name}`" for name in UNKNOWN_FIELDS),
                "",
                "## Old exploratory manifest",
                "",
                f"- old records: {len(old_paths)}",
                f"- new-only records: {len(old_missing)}",
                f"- old-only records: {len(old_extra)}",
                "",
            ]
        )
        if old_missing_details:
            report.extend(["New-only records restored by this audit:", ""])
            for item in old_missing_details:
                report.append(
                    f"- `{item['record_id']}` `{item['raw_member']}` "
                    f"({item['label']}, {item['condition_id']})"
                )
            report.extend(
                [
                    "",
                    "These four KA04 records were omitted by the old manifest without a "
                    "recorded reason. They pass byte-hash and deep signal checks and are "
                    "kept in the new manifest. Any later source-level exclusion must be "
                    "documented explicitly.",
                    "",
                ]
            )
        if errors:
            report.extend(["## Errors", ""] + [f"- {item}" for item in errors[:50]])
        if warnings:
            report.extend(["## Warnings", ""] + [f"- {item}" for item in warnings[:50]])
        (OUT / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"status: {'PASS' if not errors else 'FAIL'}")
    print(f"selected records: {len(rows)}")
    print(f"errors: {len(errors)} warnings: {len(warnings)}")
    if errors:
        for error in errors[:30]:
            print("ERROR:", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
