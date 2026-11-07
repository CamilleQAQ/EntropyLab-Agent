"""Paderborn: hold out whole physical bearings, not measurements."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.io.dataset import load_dataset_spec  # noqa: E402
from entropy_agent.training.pipeline import (  # noqa: E402
    evaluate_partition,
    fit_prepared,
    prepare_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "configs/paderborn_3class_fusion_time_extratrees.json"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/paper/paderborn_3class")
    parser.add_argument("--output", type=Path, default=ROOT / "results/paderborn-bearing-split-20261107")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    spec = load_dataset_spec(args.spec)
    prepared = prepare_dataset(spec, args.data_root)
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    bearings = np.asarray(
        [entry["bearing"] for cls in manifest["classes"] for entry in cls["files"]],
        dtype=object,
    )
    if len(bearings) != len(prepared.labels):
        raise SystemExit(f"bearing map {len(bearings)} != records {len(prepared.labels)}")

    unique = sorted(set(bearings))
    assignment = {name: index % args.folds for index, name in enumerate(unique)}
    print(f"{len(bearings)} records, {len(unique)} bearings: {unique}")

    rows = []
    for fold in range(args.folds):
        test = np.asarray([assignment[name] == fold for name in bearings])
        view = dataclasses.replace(
            prepared,
            masks={"train": ~test, "validation": test, "test": test},
            split_by_group={},
        )
        model, _ = fit_prepared(view, spec)
        metrics = evaluate_partition(
            view, model, "test", healthy_class_id=spec.labels.healthy_class_id
        )[0]
        rows.append(
            {
                "fold": fold,
                "held_out_bearings": sorted(set(bearings[test])),
                "macro_f1": round(float(metrics["multiclass_macro_f1"]), 4),
                "accuracy": round(float(metrics["multiclass_accuracy"]), 4),
                "records": int(test.sum()),
            }
        )
        print(rows[-1])

    values = [row["macro_f1"] for row in rows]
    summary = {
        "folds": args.folds,
        "mean_macro_f1": round(float(np.mean(values)), 4),
        "std_macro_f1": round(float(np.std(values, ddof=1)), 4),
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"unseen-bearing macro-F1: {summary['mean_macro_f1']} +- {summary['std_macro_f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
