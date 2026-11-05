#!/usr/bin/env python
"""Run the v1.1 Agent and its same-budget controls on one dataset.

The dataset is injected through a ``DatasetSpec`` file; this script holds no
dataset-specific logic.  Every arm is repeated over several split seeds, and the
test partition is scored once, for the champion only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.agent import run_agent  # noqa: E402
from entropy_agent.io.dataset import load_dataset_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True, help="DatasetSpec JSON")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per-round", type=int, default=3)
    parser.add_argument(
        "--units",
        type=int,
        default=None,
        help=(
            "v1.2 budget: how many entropy methods to pay a feature extraction "
            "for; each paid unit evaluates all of its classifiers. Overrides "
            "--rounds/--per-round."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="20261104",
        help="comma separated split seeds, for example 20261104,1,2,3,4",
    )
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=3,
        help="group-isolated validation splits averaged per trial",
    )
    parser.add_argument(
        "--arms",
        default="agent,random,fixed,exhaustive",
        help="comma separated subset of agent,random,fixed,exhaustive",
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--memory", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    spec = load_dataset_spec(args.spec)
    summary = run_agent(
        spec,
        args.data_root,
        args.output,
        rounds=args.rounds,
        per_round=args.per_round,
        units=args.units,
        seeds=seeds,
        validation_repeats=args.validation_repeats,
        arms=tuple(part.strip() for part in args.arms.split(",") if part.strip()),
        cache_root=args.cache_root,
        memory_path=args.memory,
        model_id=args.model_id,
        run_test=not args.no_test,
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    if summary["champion"]:
        print(f"champion: {summary['champion']['method']} + {summary['champion']['classifier']}")
        print(f"champion validation: {summary['champion_validation']}")
    if summary["test"]:
        test = summary["test"]["test"]
        print(
            f"test (single shot): accuracy={test['multiclass_accuracy']:.4f} "
            f"macro_f1={test['multiclass_macro_f1']:.4f}"
        )
    print(f"report -> {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
