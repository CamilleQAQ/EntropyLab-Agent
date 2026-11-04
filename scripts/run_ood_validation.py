#!/usr/bin/env python
"""Validate the out-of-distribution gate with a leave-one-class-out protocol.

For every class of the dataset:

1. train the configured feature + classifier on the *other* classes;
2. calibrate the OOD threshold on that fold's validation split only
   (``VALIDATION_MAX_NEAREST_TRAINING``, margin 1.0 so the raw threshold is kept);
3. score the fold's known test records (in-distribution) and every record of the
   held-out class (never seen by the fold's model).

A record is rejected at margin ``m`` when the strict majority of its windows are
farther from the training anchors than ``threshold * m``.  Because distances and
the raw threshold are stored per record, the margin sweep costs nothing after the
single pass.

The real diagnosis entry point ``diagnose_artifact_signal`` is exercised on a
sample of records per fold and its decision is compared with the vectorized
replica; any disagreement fails the run.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_agent.diagnosis import diagnose_artifact_signal, load_model_artifact
from entropy_agent.io.dataset import (
    DatasetRecord,
    DatasetSpec,
    load_dataset_records,
    load_record_signal,
)
from entropy_agent.io.dataset import OODConfig
from entropy_agent.training.grouped import stratified_group_split
from entropy_agent.training.pipeline import (
    PreparedDataset,
    extract_features,
    train_dataset,
)

DEFAULT_MARGINS = (1.0, 1.05, 1.1, 1.2, 1.5)


def _prepared(
    spec: DatasetSpec,
    records: list[DatasetRecord],
    features: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    split_by_group: dict[str, str],
) -> PreparedDataset:
    groups = tuple(record.group for record in records)
    split = np.asarray([split_by_group[group] for group in groups])
    masks = {name: split == name for name in ("train", "validation", "test")}
    return PreparedDataset(
        spec=spec,
        records=records,
        features=features,
        labels=labels,
        groups=groups,
        feature_dimension=int(dimension),
        split_by_group=split_by_group,
        masks=masks,
        feature_seconds=0.0,
    )


def _distances(
    features: np.ndarray, center: np.ndarray, scale: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Nearest-anchor standardized distance for every record (mean over windows)."""
    flattened = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    standardized = (flattened - center) / scale
    distances = np.min(
        np.linalg.norm(anchors[None, :, :] - standardized[:, None, :], axis=2), axis=1
    )
    return distances.reshape(features.shape[0], -1)


def _rejected(window_distances: np.ndarray, threshold: float, margin: float) -> bool:
    outliers = int((window_distances > threshold * margin).sum())
    return outliers > window_distances.size // 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins", default=",".join(str(value) for value in DEFAULT_MARGINS))
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--verify-per-fold",
        type=int,
        default=4,
        help="records per fold also sent through the real diagnosis runner",
    )
    args = parser.parse_args()

    margins = tuple(float(part) for part in args.margins.split(",") if part.strip())
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    spec = load_dataset_spec_with_ood(args.spec)
    records = load_dataset_records(spec, args.data_root)
    labels_all = np.asarray([record.label for record in records], dtype=np.int64)
    groups_all = [record.group for record in records]
    classes = sorted(set(int(label) for label in labels_all))

    print(f"records={len(records)} classes={classes}; extracting features once ...", flush=True)
    _, features_all, _, dimension = extract_features(spec, args.data_root, records=records)
    print(f"features {features_all.shape}; running {len(classes)} folds", flush=True)

    rows: list[dict] = []
    verification_failures: list[str] = []
    for held_out in classes:
        keep = labels_all != held_out
        kept_classes = [value for value in classes if value != held_out]
        # The fold model only knows the kept classes, so the fold spec declares
        # exactly those (renumbered from zero) or the artifact refuses to load.
        label_map = {value: index for index, value in enumerate(kept_classes)}
        healthy = spec.labels.healthy_class_id
        fold_spec = spec.model_copy(
            update={
                "labels": spec.labels.model_copy(
                    update={
                        "names": tuple(spec.labels.names[value] for value in kept_classes),
                        "mapping": {
                            spec.labels.names[value]: index
                            for index, value in enumerate(kept_classes)
                        },
                        "healthy_class_id": (
                            label_map[healthy] if healthy in label_map else None
                        ),
                    }
                )
            }
        )
        fold_records = [record for index, record in enumerate(records) if keep[index]]
        fold_labels = np.asarray(
            [label_map[int(labels_all[index])] for index in range(len(records)) if keep[index]],
            dtype=np.int64,
        )
        fold_features = features_all[keep]
        fold_groups = [groups_all[index] for index in range(len(records)) if keep[index]]
        split_by_group = stratified_group_split(
            fold_labels,
            fold_groups,
            seed=args.seed,
            train_fraction=fold_spec.split.train_fraction,
            validation_fraction=fold_spec.split.validation_fraction,
        )
        prepared = _prepared(
            fold_spec, fold_records, fold_features, fold_labels, dimension, split_by_group
        )
        model_directory = output / f"fold-{held_out}"
        print(f"fold held_out={held_out}: training ...", flush=True)
        train_dataset(
            fold_spec,
            args.data_root,
            model_directory,
            model_id=f"{fold_spec.dataset_id.lower()}-ood-fold-{held_out}",
            prepared=prepared,
            evaluate_test=False,
        )
        artifact = load_model_artifact(model_directory, trusted=True)
        ood = artifact.config["ood"]
        if not ood["enabled"]:
            raise ValueError("OOD contract was not calibrated")
        center = np.asarray(ood["center"], dtype=np.float64)
        scale = np.asarray(ood["scale"], dtype=np.float64)
        anchors = np.asarray(ood["anchors"], dtype=np.float64)
        threshold = float(ood["threshold"])

        raw_predictions = np.asarray(
            artifact.model.predict(
                np.asarray(features_all, dtype=np.float64).reshape(-1, dimension)
            )
        ).reshape(len(records))
        # Map fold-local class ids back to the dataset's class ids.
        predictions = np.asarray([kept_classes[int(value)] for value in raw_predictions])
        distances = _distances(np.asarray(features_all), center, scale, anchors)

        known_indices = np.flatnonzero(
            keep
            & np.asarray(
                [split_by_group.get(record.group) == "test" for record in records]
            )
        )
        unknown_indices = np.flatnonzero(labels_all == held_out)
        for role, indices in (("known", known_indices), ("unknown", unknown_indices)):
            for index in indices:
                rows.append(
                    {
                        "fold": held_out,
                        "role": role,
                        "record_id": records[index].record_id,
                        "truth_class": int(labels_all[index]),
                        "predicted_class": int(predictions[index]),
                        "threshold": threshold,
                        "window_count": int(distances.shape[1]),
                        "window_distances": distances[index].tolist(),
                    }
                )

        # Exercise the real entry point on a small sample and compare decisions.
        sample_indices = list(known_indices[: args.verify_per_fold]) + list(
            unknown_indices[: args.verify_per_fold]
        )
        for index in sample_indices:
            report = diagnose_artifact_signal(
                load_record_signal(records[index]).values,
                model_directory,
                sampling_rate_hz=spec.sampling_rate_hz,
                trusted=True,
            )
            runner_rejected = report["status"] != "DIAGNOSED"
            window_distances = [
                float(window["ood_distance"]) for window in report.get("window_predictions", [])
            ]
            if len(window_distances) != distances.shape[1]:
                verification_failures.append(
                    f"fold {held_out} {records[index].record_id}: window count mismatch"
                )
                continue
            replica_rejected = _rejected(np.asarray(window_distances), threshold, 1.0)
            if runner_rejected != replica_rejected:
                verification_failures.append(
                    f"fold {held_out} {records[index].record_id}: "
                    f"runner={report['status']} replica_rejected={replica_rejected}"
                )
            if not runner_rejected and int(report["class_id"]) != int(raw_predictions[index]):
                verification_failures.append(
                    f"fold {held_out} {records[index].record_id}: class mismatch"
                )
        print(f"fold held_out={held_out}: done", flush=True)

    summary = _summarise(rows, margins, classes)
    _write_rows(output / "records.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_id": spec.dataset_id,
                "feature": spec.feature.method,
                "classifier": spec.classifier.algorithm,
                "classes": classes,
                "margins": list(margins),
                "runner_verification": {
                    "checked": len(classes) * 2 * args.verify_per_fold,
                    "failures": verification_failures,
                },
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output / "report.md", spec, summary, verification_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if verification_failures:
        raise SystemExit(f"{len(verification_failures)} runner verification failures")
    print(f"report -> {output / 'report.md'}")
    return 0


def load_dataset_spec_with_ood(path: Path) -> DatasetSpec:
    """Load the spec and force OOD calibration with margin 1.0 (raw threshold)."""
    from entropy_agent.io.dataset import load_dataset_spec

    spec = load_dataset_spec(path)
    return spec.model_copy(
        update={"ood": OODConfig(calibration="VALIDATION_MAX_NEAREST_TRAINING", margin=1.0)}
    )


def _summarise(rows: list[dict], margins: tuple[float, ...], classes: list[int]) -> dict:
    per_margin: dict[str, dict] = {}
    for margin in margins:
        known_total = known_rejected = known_accepted = known_correct = 0
        unknown_total = unknown_accepted = 0
        for row in rows:
            rejected = _rejected(
                np.asarray(row["window_distances"], dtype=np.float64),
                float(row["threshold"]),
                margin,
            )
            if row["role"] == "known":
                known_total += 1
                if rejected:
                    known_rejected += 1
                else:
                    known_accepted += 1
                    known_correct += int(row["predicted_class"] == row["truth_class"])
            else:
                unknown_total += 1
                unknown_accepted += int(not rejected)
        per_margin[f"{margin:.2f}"] = {
            "known_total": known_total,
            "known_accepted": known_accepted,
            "known_rejected": known_rejected,
            "false_rejection_rate": known_rejected / known_total if known_total else None,
            "known_accuracy_among_accepted": (
                known_correct / known_accepted if known_accepted else None
            ),
            "known_accuracy_overall": known_correct / known_total if known_total else None,
            "unknown_total": unknown_total,
            "unknown_accepted": unknown_accepted,
            "false_acceptance_rate": (
                unknown_accepted / unknown_total if unknown_total else None
            ),
        }
    return {
        "classes": classes,
        "known_records": sum(1 for row in rows if row["role"] == "known"),
        "unknown_records": sum(1 for row in rows if row["role"] == "unknown"),
        "per_margin": per_margin,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["fold", "role", "record_id", "truth_class", "predicted_class", "threshold",
             "window_count", "max_window_distance"]
        )
        for row in rows:
            writer.writerow(
                [row["fold"], row["role"], row["record_id"], row["truth_class"],
                 row["predicted_class"], row["threshold"], row["window_count"],
                 max(row["window_distances"])]
            )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _write_report(
    path: Path, spec: DatasetSpec, summary: dict, failures: list[str]
) -> None:
    lines = ["# OOD 验证报告（留一类）", ""]
    lines.append(f"- 数据集：`{spec.dataset_id}`")
    lines.append(f"- 特征 / 分类器：`{spec.feature.method}` / `{spec.classifier.algorithm}`")
    lines.append(f"- 已知记录：{summary['known_records']}，未知记录：{summary['unknown_records']}")
    lines.append("- 阈值：仅用各折 validation 校准（margin=1.0 保存原始阈值），扫描在记录级离线完成")
    lines.append("")
    lines.append("| margin | 误拒率 FRR | 已知类接受后准确率 | 已知类整体准确率 | 误接受率 FAR |")
    lines.append("|---:|---:|---:|---:|---:|")
    for margin, stats in summary["per_margin"].items():
        lines.append(
            f"| {margin} | {_pct(stats['false_rejection_rate'])} | "
            f"{_pct(stats['known_accuracy_among_accepted'])} | "
            f"{_pct(stats['known_accuracy_overall'])} | "
            f"{_pct(stats['false_acceptance_rate'])} |"
        )
    lines.append("")
    lines.append(f"- 真实诊断入口校验：{len(failures)} 处不一致")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
