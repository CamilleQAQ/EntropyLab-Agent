"""Nested CV on Paderborn: naive winner's curse vs nested selection."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from entropy_agent.agent.candidates import build_pool  # noqa: E402
from entropy_agent.continual.selection import candidate_spec  # noqa: E402
from entropy_agent.io.dataset import (  # noqa: E402
    load_dataset_records,
    load_dataset_spec,
    resolve_dataset_split,
)
from entropy_agent.training.pipeline import (  # noqa: E402
    evaluate_partition,
    fit_prepared,
    prepare_dataset,
)

FEATURES = {"FUSION_TIME", "FUSION_TIME_SPEC", "FUSION_TIME_ENVELOPE", "MDE", "MPE", "RCMDE"}
CLASSIFIERS = {"EXTRA-TREES", "SVM-RBF", "RANDOM-FOREST"}


def view(prepared, groups, train_groups, query_groups, which):
    train = np.isin(groups, list(train_groups))
    query = np.isin(groups, list(query_groups))
    masks = {"train": train, "validation": query, "test": query}
    mapping = {group: ("train" if group in train_groups else "test") for group in set(groups)}
    mapping.update({group: which for group in query_groups})
    return dataclasses.replace(prepared, masks=masks, split_by_group=mapping)


def score(frame, spec, groups, train_groups, query_groups, which):
    prepared = view(frame["prepared"], groups, train_groups, query_groups, which)
    model, _ = fit_prepared(prepared, frame["spec"])
    metrics = evaluate_partition(
        prepared, model, which, healthy_class_id=spec.labels.healthy_class_id
    )[0]
    key = "multiclass_macro_f1" if "multiclass_macro_f1" in metrics else "multiclass_f1"
    return float(metrics[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper/paderborn_3class")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nested-cv-20261107")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--outer", type=int, default=3)
    parser.add_argument("--inner", type=int, default=3)
    args = parser.parse_args()

    spec = load_dataset_spec(args.spec)
    records = load_dataset_records(spec, args.data_root)
    labels = np.asarray([record.label for record in records])
    groups = np.asarray([record.group for record in records])

    frames = []
    for candidate in build_pool():
        if candidate.method.method not in FEATURES or candidate.classifier.algorithm not in CLASSIFIERS:
            continue
        seeded = candidate_spec(spec, candidate)
        frames.append(
            {"candidate": candidate, "spec": seeded, "prepared": prepare_dataset(seeded, args.data_root)}
        )
    print(f"{len(frames)} candidates, {len(records)} records, {len(set(groups))} groups")

    naive = []
    for offset in range(args.seeds):
        seeded = spec.model_copy(update={"split": spec.split.model_copy(update={"seed": spec.split.seed + offset})})
        assignment = resolve_dataset_split(seeded, records)
        names = np.asarray([assignment[group] for group in groups])
        train_groups = set(groups[names == "train"])
        val_groups = set(groups[names == "validation"])
        test_groups = set(groups[names == "test"])
        best = None
        for frame in frames:
            value = score(frame, spec, groups, train_groups, val_groups, "validation")
            if best is None or value > best[0]:
                best = (value, frame)
        test_value = score(best[1], spec, groups, train_groups, test_groups, "test")
        naive.append(
            {
                "seed": offset,
                "selected": best[1]["candidate"].candidate_id.split("@")[0],
                "validation": round(best[0], 4),
                "test": round(test_value, 4),
                "gap": round(best[0] - test_value, 4),
            }
        )
        print("naive", naive[-1])

    nested = []
    outer = StratifiedGroupKFold(n_splits=args.outer, shuffle=True, random_state=spec.split.seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(labels)), labels, groups)):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        inner = StratifiedGroupKFold(n_splits=args.inner, shuffle=True, random_state=spec.split.seed + fold + 1)
        inner_labels, inner_groups = labels[train_idx], groups[train_idx]
        inner_splits = list(inner.split(np.zeros(len(inner_labels)), inner_labels, inner_groups))
        best = None
        for frame in frames:
            values = [
                score(frame, spec, groups, set(inner_groups[a]), set(inner_groups[b]), "validation")
                for a, b in inner_splits
            ]
            mean = float(np.mean(values))
            if best is None or mean > best[0]:
                best = (mean, frame)
        test_value = score(best[1], spec, groups, train_groups, test_groups, "test")
        nested.append(
            {
                "fold": fold,
                "selected": best[1]["candidate"].candidate_id.split("@")[0],
                "inner": round(best[0], 4),
                "outer_test": round(test_value, 4),
                "gap": round(best[0] - test_value, 4),
            }
        )
        print("nested", nested[-1])

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "naive": naive,
        "nested": nested,
        "naive_mean_gap": round(float(np.mean([row["gap"] for row in naive])), 4),
        "nested_mean_gap": round(float(np.mean([row["gap"] for row in nested])), 4),
        "naive_mean_test": round(float(np.mean([row["test"] for row in naive])), 4),
        "nested_mean_test": round(float(np.mean([row["outer_test"] for row in nested])), 4),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if not isinstance(value, list)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
