#!/usr/bin/env python
"""Run the v1 Agent (5 rounds) and its same-budget controls on one dataset.

The dataset is injected through a ``DatasetSpec`` file; this script holds no
dataset-specific logic.  Test is opened once, for the champion only.
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
    parser.add_argument("--seed", type=int, default=20261104)
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

    spec = load_dataset_spec(args.spec)
    summary = run_agent(
        spec,
        args.data_root,
        args.output,
        rounds=args.rounds,
        per_round=args.per_round,
        seed=args.seed,
        arms=tuple(part.strip() for part in args.arms.split(",") if part.strip()),
        cache_root=args.cache_root,
        memory_path=args.memory,
        model_id=args.model_id,
        run_test=not args.no_test,
    )
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
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
