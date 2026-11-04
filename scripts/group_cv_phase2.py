#!/usr/bin/env python
"""Leave-one-bearing-out diagnostic on the 12 development bearings.

This does not touch final_holdout.  It answers whether the weak validation
score is split-specific or a feature-generalisation problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from entropy_agent.io.dataset import load_dataset_spec
from entropy_agent.training.pipeline import prepare_dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paderborn_fusion_baseline.json"
OUT = ROOT / "results" / "phase2-groupcv-20261104"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "features.npz"
    spec = load_dataset_spec(CONFIG)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        x = data["features"]
        y = data["labels"]
        groups = data["groups"].astype(str)
        print("loaded feature cache", cache)
    else:
        prepared = prepare_dataset(spec, ROOT, include_test=False)
        x = prepared.features.reshape(len(prepared.records), -1)
        y = prepared.labels
        groups = np.asarray([record.group for record in prepared.records], dtype=str)
        np.savez_compressed(cache, features=x, labels=y, groups=groups)
        print("wrote feature cache", cache)
    unique_groups = sorted(set(groups.tolist()))
    print("records", x.shape, "groups", unique_groups)

    all_truth = []
    all_pred = []
    per_group = {}
    for group in unique_groups:
        test_mask = groups == group
        train_mask = ~test_mask
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=0,
            n_jobs=1,
        )
        model.fit(x[train_mask], y[train_mask])
        predicted = model.predict(x[test_mask])
        truth = y[test_mask]
        all_truth.extend(truth.tolist())
        all_pred.extend(predicted.tolist())
        accuracy = float(accuracy_score(truth, predicted))
        per_group[str(group)] = {
            "records": int(test_mask.sum()),
            "accuracy": accuracy,
            "confusion": confusion_matrix(
                truth, predicted, labels=range(len(spec.labels.names))
            ).tolist(),
        }
        print(group, "accuracy", round(accuracy, 4), "n", int(test_mask.sum()))

    truth_array = np.asarray(all_truth)
    pred_array = np.asarray(all_pred)
    result = {
        "config": str(CONFIG),
        "groups": unique_groups,
        "records": int(len(truth_array)),
        "pooled_accuracy": float(accuracy_score(truth_array, pred_array)),
        "pooled_macro_f1": float(
            f1_score(
                truth_array,
                pred_array,
                labels=range(len(spec.labels.names)),
                average="macro",
                zero_division=0,
            )
        ),
        "pooled_confusion": confusion_matrix(
            truth_array, pred_array, labels=range(len(spec.labels.names))
        ).tolist(),
        "per_group": per_group,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("pooled accuracy", round(result["pooled_accuracy"], 4))
    print("pooled macro_f1", round(result["pooled_macro_f1"], 4))
    print("confusion", result["pooled_confusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
