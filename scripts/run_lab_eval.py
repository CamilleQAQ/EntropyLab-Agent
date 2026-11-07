#!/usr/bin/env python
"""Run the task-level evaluation set for the research lab."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.io.dataset import load_dataset_spec  # noqa: E402
from entropy_agent.lab import LabContext, chat, run_lab  # noqa: E402
from entropy_agent.lab.eval import load_tasks, score  # noqa: E402

DATASETS = {
    "aero": ("configs/aero_5state_fusion_time_extratrees.json", "data/paper/aero_engine_5state"),
    "triaxial": ("configs/triaxial_7class_tsmefude_btlssvm.json", "data/paper/triaxial_7class"),
    "gear": ("configs/gear_5state_fusion_time_extratrees.json", "data/paper/gear_5state"),
    "vbl": ("configs/vbl_4class_fusion_time_extratrees.json", "data/paper/vbl_4class"),
    "paderborn": ("configs/paderborn_3class_fusion_time_extratrees.json", "data/paper/paderborn_3class"),
    "cwru": ("configs/cwru_4class_fusion_time_extratrees.json", "data/paper/cwru_4class"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs/lab_tasks.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lab-eval-20261107")
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--only", default=None, help="comma separated task ids")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        tasks = [task for task in tasks if task.task_id in wanted]

    rows = []
    for task in tasks:
        config, data = DATASETS[task.dataset]
        spec = load_dataset_spec(ROOT / config)
        workspace = args.output / task.task_id
        ctx = LabContext(
            spec=spec,
            data_root=ROOT / data,
            workspace=workspace,
            goal=task.goal,
            budget_seconds=args.budget,
        )
        try:
            summary = run_lab(ctx, goal=task.goal, max_rounds=task.max_rounds, llm=chat)
        except Exception as error:  # a failed task must not stop the suite
            rows.append(
                {
                    "task": task.task_id,
                    "dataset": task.dataset,
                    "score": 0.0,
                    "passed": 0,
                    "total": 9,
                    "failed": f"ERROR: {type(error).__name__}: {error}"[:160],
                }
            )
            print(f"{task.task_id:24s} ERROR {type(error).__name__}: {error}")
            continue

        result = score(task.expect, summary)
        rows.append(
            {
                "task": task.task_id,
                "dataset": task.dataset,
                "score": result["score"],
                "passed": result["passed"],
                "total": result["total"],
                "failed": ",".join(result["failed"]),
            }
        )
        print(
            f"{task.task_id:24s} {result['passed']}/{result['total']} "
            f"({result['score']:.2f}) failed={result['failed']}"
        )

    csv_path = args.output / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean_score = sum(row["score"] for row in rows) / len(rows)
    (args.output / "summary.md").write_text(
        "# 任务级评测\n\n"
        + "\n".join(
            f"- `{row['task']}`（{row['dataset']}）：{row['passed']}/{row['total']}"
            + (f"，未通过：{row['failed']}" if row["failed"] else "")
            for row in rows
        )
        + f"\n\n平均得分：**{mean_score:.2f}**\n",
        encoding="utf-8",
    )
    print(f"\nmean score {mean_score:.2f} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
