#!/usr/bin/env python
"""Generic supervised experiment runner for paper reproduction.

Unlike the Phase 2 golden-path runner, this evaluates the test partition.  It is
intended for the paper datasets where a random/group-isolated holdout is the
reported protocol.
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


def write_predictions(path: Path, prepared, model, split: str, labels: tuple[str, ...]) -> None:
    mask = prepared.masks[split]
    indices = np.flatnonzero(mask)
    _, record_predictions, window_predictions, _ = evaluate_partition(
        prepared, model, split, healthy_class_id=prepared.spec.labels.healthy_class_id
    )
    rows = []
    for position, index in enumerate(indices):
        record = prepared.records[index]
        truth = int(prepared.labels[index])
        predicted = int(record_predictions[position])
        votes = np.bincount(window_predictions[position], minlength=len(labels))
        row = {
            "record_id": record.record_id,
            "group": record.group,
            "truth_id": truth,
            "truth_label": labels[truth],
            "predicted_id": predicted,
            "predicted_label": labels[predicted],
            "window_count": int(window_predictions.shape[1]),
        }
        # One vote column per configured class keeps multiclass records auditable.
        row.update(
            {f"vote_{index}": int(votes[index]) for index in range(len(labels))}
        )
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    out = args.output
    model_dir = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        print(f"model directory already exists, refusing to overwrite: {model_dir}")
        return 2

    spec = load_dataset_spec(args.config)
    prepared = prepare_dataset(spec, args.data_root, include_test=True)
    result = train_dataset(
        spec,
        args.data_root,
        model_dir,
        model_id=args.model_id,
        prepared=prepared,
        evaluate_test=True,
    )
    labels = tuple(spec.labels.names)
    write_predictions(out / "validation_predictions.csv", prepared, result.artifact.model, "validation", labels)
    write_predictions(out / "test_predictions.csv", prepared, result.artifact.model, "test", labels)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "data_root": str(args.data_root),
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "feature": {
            "method": spec.feature.method,
            "parameters": spec.feature.parameters,
            "dimension": prepared.feature_dimension,
        },
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
            "seed": spec.split.seed,
            "records": {name: int(mask.sum()) for name, mask in prepared.masks.items()},
            "groups": {
                name: len(
                    {
                        group
                        for group, value in prepared.split_by_group.items()
                        if value == name
                    }
                )
                for name in ("train", "validation", "test")
            },
        },
        "validation": result.test_result["validation"],
        "test": result.test_result["test"],
        "timings": result.test_result["timings"],
        "total_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("experiment complete")
    print(f"feature_dimension={prepared.feature_dimension}")
    print(f"records={summary['split']['records']}")
    print(f"validation_accuracy={summary['validation']['multiclass_accuracy']:.6f}")
    print(f"validation_macro_f1={summary['validation']['multiclass_macro_f1']:.6f}")
    print(f"test_accuracy={summary['test']['multiclass_accuracy']:.6f}")
    print(f"test_macro_f1={summary['test']['multiclass_macro_f1']:.6f}")
    print(f"model={model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
