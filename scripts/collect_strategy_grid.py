#!/usr/bin/env python
"""Collect (dataset, feature, classifier) -> validation macro-F1 across datasets.

The output feeds the strategy learner: it answers "given a dataset profile and a
candidate, what score should I expect?"  Every dataset uses its own spec, so the
numbers are the same ones the CLI would report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.io.dataset import load_dataset_spec  # noqa: E402
from entropy_agent.lab.tools import LabContext, RunExperimentArgs, run_experiment  # noqa: E402

DATASETS = (
    ("aero", "configs/aero_5state_fusion_time_extratrees.json", "data/paper/aero_engine_5state"),
    ("triaxial", "configs/triaxial_7class_tsmefude_btlssvm.json", "data/paper/triaxial_7class"),
    ("gear", "configs/gear_5state_fusion_time_extratrees.json", "data/paper/gear_5state"),
    ("vbl", "configs/vbl_4class_fusion_time_extratrees.json", "data/paper/vbl_4class"),
    ("paderborn", "configs/paderborn_3class_fusion_time_extratrees.json", "data/paper/paderborn_3class"),
    ("cwru", "configs/cwru_4class_fusion_time_extratrees.json", "data/paper/cwru_4class"),
)

FEATURES = ("FUSION_TIME", "FUSION_TIME_SPEC", "FUSION_TIME_ENVELOPE", "MDE", "MPE", "RCMDE")
CLASSIFIERS = ("EXTRA-TREES", "SVM-RBF", "RANDOM-FOREST")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/strategy-20261106/grid.csv")
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, config, data in DATASETS:
        spec = load_dataset_spec(ROOT / config)
        ctx = LabContext(
            spec=spec,
            data_root=ROOT / data,
            workspace=args.output.parent / f"runs-{name}",
            budget_seconds=3600,
        )
        for feature in FEATURES:
            for classifier in CLASSIFIERS:
                try:
                    result = run_experiment(
                        ctx,
                        RunExperimentArgs(feature=feature, classifier=classifier, seeds=args.seeds),
                    )
                except Exception as error:  # keep the grid going
                    print(f"{name} {feature}+{classifier} FAILED: {type(error).__name__}: {error}")
                    continue
                for entry in result["per_seed"]:
                    rows.append(
                        {
                            "dataset": name,
                            "feature": feature,
                            "classifier": classifier,
                            "seed": entry["seed"],
                            "macro_f1": entry["macro_f1"],
                            "accuracy": entry["accuracy"],
                            "seconds": round(result["seconds"] / max(len(result["per_seed"]), 1), 3),
                        }
                    )
                print(
                    f"{name:9s} {feature:24s} {classifier:14s} "
                    f"{result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
                )

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
