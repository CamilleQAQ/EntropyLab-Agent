#!/usr/bin/env python
"""Phase 2 golden path runner for a DatasetSpec configuration.

Only initial_train and initial_validation are loaded.  final_holdout is never
passed to this script and its groups are checked as absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_agent.io.dataset import load_dataset_spec
from entropy_agent.training.pipeline import (
    evaluate_partition,
    prepare_dataset,
    train_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_groups() -> set[str]:
    groups: set[str] = set()
    with (ROOT / "manifests" / "paderborn" / "final_holdout.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            groups.add(row["group_id"])
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "paderborn_baseline.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "phase2-baseline-20261104",
    )
    parser.add_argument("--model-id", default="paderborn-baseline-mde-svm")
    args = parser.parse_args()

    started = time.perf_counter()
    config = args.config
    out = args.output
    model_dir = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        print(f"model directory already exists, refusing to overwrite: {model_dir}")
        return 2

    spec = load_dataset_spec(config)
    if spec.split.expected_records != {"train": 640, "validation": 320, "test": 0}:
        raise ValueError(f"unexpected split contract: {spec.split.expected_records}")

    prepared = prepare_dataset(spec, ROOT, include_test=False)
    if int(prepared.masks["test"].sum()) != 0:
        raise ValueError("prepared dataset unexpectedly contains test records")
    if set(record.group for record in prepared.records) & final_groups():
        raise ValueError("development data overlaps final_holdout groups")

    result = train_dataset(
        spec,
        ROOT,
        model_dir,
        model_id=args.model_id,
        prepared=prepared,
        evaluate_test=False,
    )
    if result.test_result.get("test_evaluation_status") != "DEFERRED":
        raise ValueError("test evaluation was not deferred")

    metrics, record_predictions, window_predictions, inference_seconds = evaluate_partition(
        prepared,
        result.artifact.model,
        "validation",
        healthy_class_id=spec.labels.healthy_class_id,
    )
    if result.test_result["validation"] != metrics:
        raise ValueError("validation metrics differ between training and reloaded evaluation")

    mask = prepared.masks["validation"]
    indices = np.flatnonzero(mask)
    labels = np.asarray([name for name in spec.labels.names])
    rows = []
    for position, index in enumerate(indices):
        record = prepared.records[index]
        truth = int(prepared.labels[index])
        predicted = int(record_predictions[position])
        votes = np.bincount(window_predictions[position], minlength=len(labels))
        rows.append(
            {
                "record_id": record.record_id,
                "group": record.group,
                "truth_id": truth,
                "truth_label": labels[truth],
                "predicted_id": predicted,
                "predicted_label": labels[predicted],
                "window_count": int(window_predictions.shape[1]),
                "vote_0": int(votes[0]),
                "vote_1": int(votes[1]),
                "vote_2": int(votes[2]),
            }
        )
    with (out / "validation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_by_group = prepared.split_by_group
    training_rows = int(
        prepared.features[prepared.masks["train"]].reshape(-1, prepared.feature_dimension).shape[0]
    )
    validation_rows = int(
        prepared.features[prepared.masks["validation"]].reshape(-1, prepared.feature_dimension).shape[0]
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "config": str(config),
        "config_sha256": sha256(config),
        "manifest": str(ROOT / spec.layout.manifest),
        "manifest_sha256": sha256(ROOT / spec.layout.manifest),
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "data_root": str(ROOT),
        "feature_method": spec.feature.method,
        "feature_parameters": spec.feature.parameters,
        "classifier": {
            "algorithm": spec.classifier.algorithm,
            "backend": spec.classifier.backend,
            "parameters": spec.classifier.parameters,
        },
        "window": {
            "size": spec.window.size,
            "stride": spec.window.stride,
            "windows_per_record": spec.window.windows_per_record,
        },
        "split": {
            "mode": spec.split.mode,
            "expected_records": spec.split.expected_records,
            "groups": {
                name: sorted(group for group, value in split_by_group.items() if value == name)
                for name in ("train", "validation")
            },
        },
        "feature_dimension": prepared.feature_dimension,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "validation_metrics": metrics,
        "validation_inference_seconds": inference_seconds,
        "feature_extraction_seconds": prepared.feature_seconds,
        "total_seconds": time.perf_counter() - started,
        "test_evaluation_status": "DEFERRED",
        "final_holdout_groups_present": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "validation_confusion_matrix.json").write_text(
        json.dumps(metrics["multiclass_confusion_matrix"], indent=2) + "\n",
        encoding="utf-8",
    )

    print("Phase 2 baseline complete")
    print(f"feature_dimension={prepared.feature_dimension}")
    print(f"training_rows={training_rows} validation_rows={validation_rows}")
    print(f"validation_accuracy={metrics['multiclass_accuracy']:.6f}")
    print(f"validation_macro_f1={metrics['multiclass_macro_f1']:.6f}")
    print(f"confusion={metrics['multiclass_confusion_matrix']}")
    print(f"model={model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
