"""One supervised experiment run, shared by the CLI and script entry points.

This is the single implementation of "train on a DatasetSpec, evaluate validation,
open the test partition once, write predictions and a metrics summary".  The CLI
and `scripts/run_supervised_experiment.py` both call it, so the two entry points
cannot drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..io.dataset import DatasetSpec
from .pipeline import evaluate_partition, prepare_dataset, train_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_predictions(
    path: Path, prepared, model, split: str, labels: tuple[str, ...]
) -> None:
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
        row.update({f"vote_{index}": int(votes[index]) for index in range(len(labels))})
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_supervised_experiment(
    spec: DatasetSpec,
    data_root: Path,
    output: Path,
    *,
    model_id: str,
    config_path: Path | None = None,
) -> dict:
    """Train, evaluate once, and write metrics/predictions under ``output``."""
    started = time.perf_counter()
    output = Path(output)
    model_dir = output / "model"
    output.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        raise FileExistsError(f"model directory already exists: {model_dir}")

    prepared = prepare_dataset(spec, Path(data_root), include_test=True)
    result = train_dataset(
        spec,
        Path(data_root),
        model_dir,
        model_id=model_id,
        prepared=prepared,
        evaluate_test=True,
    )
    labels = tuple(spec.labels.names)
    write_predictions(output / "validation_predictions.csv", prepared, result.artifact.model, "validation", labels)
    write_predictions(output / "test_predictions.csv", prepared, result.artifact.model, "test", labels)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": spec.dataset_id,
        "config": str(config_path) if config_path is not None else None,
        "config_sha256": sha256(config_path) if config_path is not None else None,
        "data_root": str(data_root),
        "model_id": model_id,
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
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
