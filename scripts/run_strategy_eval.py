#!/usr/bin/env python
"""Evaluate the cross-dataset strategy on the collected grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec  # noqa: E402
from entropy_agent.lab.strategy import (  # noqa: E402
    DatasetProfile,
    aggregate,
    evaluate,
    load_grid,
    seconds_of,
)

DATASETS = {
    "aero": ("configs/aero_5state_fusion_time_extratrees.json", "data/paper/aero_engine_5state"),
    "triaxial": ("configs/triaxial_7class_tsmefude_btlssvm.json", "data/paper/triaxial_7class"),
    "gear": ("configs/gear_5state_fusion_time_extratrees.json", "data/paper/gear_5state"),
    "vbl": ("configs/vbl_4class_fusion_time_extratrees.json", "data/paper/vbl_4class"),
    "paderborn": ("configs/paderborn_3class_fusion_time_extratrees.json", "data/paper/paderborn_3class"),
    "cwru": ("configs/cwru_4class_fusion_time_extratrees.json", "data/paper/cwru_4class"),
}


def profile(name: str) -> DatasetProfile:
    config, data = DATASETS[name]
    spec = load_dataset_spec(ROOT / config)
    records = load_dataset_records(spec, ROOT / data)
    labels = [record.label for record in records]
    counts = [labels.count(index) for index in range(len(spec.labels.names))]
    return DatasetProfile(
        records=len(records),
        classes=len(spec.labels.names),
        sampling_rate=float(spec.sampling_rate_hz),
        window=int(spec.window.size),
        imbalance=round(max(counts) / max(min(counts), 1), 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=ROOT / "results/strategy-20261106/grid.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/strategy-20261106/summary.json")
    args = parser.parse_args()

    rows = load_grid(args.grid)
    table = aggregate(rows)
    seconds = seconds_of(rows)
    profiles = {name: profile(name) for name in table}

    report = evaluate(table, profiles, seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"profiles": {k: vars(v) for k, v in profiles.items()}, "report": report},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{'dataset':11s} {'best':>6s} {'fixed@1':>8s} {'learned@1':>10s} {'random@1':>9s} {'learned_top1':>40s}")
    for name, entry in report.items():
        print(
            f"{name:11s} {entry['best_possible']:6.4f} {entry['fixed@1']:8.4f} "
            f"{entry['learned@1']:10.4f} {entry['random@1']:9.4f}  {entry['learned_top1']}"
        )
    mean = lambda key: sum(entry[key] for entry in report.values()) / len(report)
    print()
    for k in (1, 2, 3):
        print(
            f"mean regret@{k}: fixed {mean(f'regret_fixed@{k}'):.4f} | "
            f"learned {mean(f'regret_learned@{k}'):.4f} | "
            f"random {mean(f'regret_random@{k}'):.4f}"
        )
    print(f"\nsummary -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
