#!/usr/bin/env python
"""Diagnose the weak Phase 2 baseline before changing the protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from entropy_agent.diagnosis.artifact import load_model_artifact
from entropy_agent.training.pipeline import evaluate_partition, prepare_dataset
from entropy_agent.io.dataset import load_dataset_spec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paderborn_baseline.json"
MODEL = ROOT / "results" / "phase2-baseline-20261104" / "model"


def main() -> int:
    spec = load_dataset_spec(CONFIG)
    prepared = prepare_dataset(spec, ROOT, include_test=False)
    x = prepared.features.reshape(len(prepared.records), -1)
    y = prepared.labels
    print("features", prepared.features.shape, "x", x.shape, "classes", spec.labels.names)
    for class_id, name in enumerate(spec.labels.names):
        rows = x[y == class_id]
        print(
            name,
            "n=", len(rows),
            "mean=", np.round(rows.mean(axis=0), 4).tolist(),
            "std=", np.round(rows.std(axis=0), 4).tolist(),
        )

    artifact = load_model_artifact(MODEL, trusted=True)
    for split in ("train", "validation"):
        metrics, _, _, _ = evaluate_partition(
            prepared,
            artifact.model,
            split,
            healthy_class_id=spec.labels.healthy_class_id,
        )
        print(
            "saved_model", split,
            "accuracy=", round(metrics["multiclass_accuracy"], 4),
            "macro_f1=", round(metrics["multiclass_macro_f1"], 4),
            "confusion=", metrics["multiclass_confusion_matrix"],
        )

    candidates = {
        "svc_c1_scale_bal": make_pipeline(
            StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced")
        ),
        "svc_c0.1_scale_bal": make_pipeline(
            StandardScaler(), SVC(C=0.1, gamma="scale", class_weight="balanced")
        ),
        "svc_c10_g0.01_bal": make_pipeline(
            StandardScaler(), SVC(C=10.0, gamma=0.01, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=0
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=0
        ),
    }

    def record_metrics(model, mask):
        predicted = model.predict(x[mask])
        truths = y[mask]
        return {
            "accuracy": float(np.mean(predicted == truths)),
            "macro_f1": float(
                f1_score(
                    truths,
                    predicted,
                    labels=range(len(spec.labels.names)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion": confusion_matrix(
                truths, predicted, labels=range(len(spec.labels.names))
            ).tolist(),
        }

    train_mask = prepared.masks["train"]
    for name, model in candidates.items():
        model.fit(x[train_mask], y[train_mask])
        for split_name, mask in (
            ("train", train_mask),
            ("validation", prepared.masks["validation"]),
        ):
            metrics = record_metrics(model, mask)
            print(
                name, split_name,
                "accuracy=", round(metrics["accuracy"], 4),
                "macro_f1=", round(metrics["macro_f1"], 4),
                "confusion=", metrics["confusion"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
