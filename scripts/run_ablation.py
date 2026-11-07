#!/usr/bin/env python
"""Component ablation: what happens when grounding or the reviewer is removed?

Runs the same tasks in three conditions (full / no grounding / no reviewer),
then measures what the guardrails were actually catching:

* ungrounded hypotheses: hypotheses whose numbers do not appear in the history;
* unstable experiments: runs with < 3 splits or macro-F1 std > 0.05.

Each condition runs in a subprocess so the import-time switches take effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.lab.eval import load_tasks, score  # noqa: E402
from entropy_agent.lab.grounding import problems  # noqa: E402

CONDITIONS = {
    "full": {},
    "no_grounding": {"LAB_DISABLE_GROUNDING": "1"},
    "no_reviewer": {"LAB_DISABLE_REVIEWER": "1"},
}

DATASETS = {
    "paderborn": ("configs/paderborn_3class_fusion_time_extratrees.json", "data/paper/paderborn_3class"),
    "cwru": ("configs/cwru_4class_fusion_time_extratrees.json", "data/paper/cwru_4class"),
    "gear": ("configs/gear_5state_fusion_time_extratrees.json", "data/paper/gear_5state"),
}


def run_condition(task, condition: str, env_extra: dict, output: Path) -> dict:
    config, data = DATASETS[task.dataset]
    command = [
        sys.executable,
        "-m",
        "entropy_agent.cli",
        "lab",
        "run",
        "--spec",
        str(ROOT / config),
        "--data-root",
        str(ROOT / data),
        "--output",
        str(output),
        "--goal",
        task.goal,
        "--max-rounds",
        str(task.max_rounds),
        "--budget",
        "300",
    ]
    environment = dict(os.environ)
    environment.update(env_extra)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, env=environment, cwd=ROOT, capture_output=True, text=True)

    ledger_path = output / "ledger.sqlite"
    if not ledger_path.is_file():
        return {"task": task.task_id, "condition": condition, "error": "no ledger"}
    connection = sqlite3.connect(ledger_path)
    hypotheses = [
        {"round": row[0], "text": row[1], "evidence": row[2]}
        for row in connection.execute("SELECT round, text, evidence FROM hypotheses ORDER BY id")
    ]
    experiments = [
        {"round": row[0], "tool": row[1], "payload": json.loads(row[2])}
        for row in connection.execute("SELECT round, tool, payload FROM experiments ORDER BY id")
    ]
    reviews = [
        {"round": row[0], "verdict": row[1], "findings": json.loads(row[2])}
        for row in connection.execute("SELECT round, verdict, findings FROM reviews ORDER BY id")
    ]
    connection.close()

    ungrounded = 0
    for hypothesis in hypotheses:
        history = [
            experiment["payload"]
            for experiment in experiments
            if experiment["round"] <= hypothesis["round"]
        ]
        if problems(hypothesis["text"], history):
            ungrounded += 1
    unstable = 0
    for experiment in experiments:
        payload = experiment["payload"]
        if payload.get("tool") != "run_experiment":
            continue
        seeds = int(payload.get("seeds") or 0)
        std = float(payload.get("macro_f1_std") or 0.0)
        if seeds < 3 or std > 0.05:
            unstable += 1

    summary = {
        "status": "done",
        "report": str(output / "report.md"),
        "experiments": experiments,
        "hypotheses": hypotheses,
    }
    scored = score(task.expect, summary)
    return {
        "task": task.task_id,
        "condition": condition,
        "score": scored["score"],
        "experiments": len([e for e in experiments if e["payload"].get("tool") == "run_experiment"]),
        "hypotheses": len(hypotheses),
        "ungrounded_hypotheses": ungrounded,
        "unstable_experiments": unstable,
        "review_findings": sum(len(review["findings"]) for review in reviews),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs/lab_tasks.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ablation-20261107")
    parser.add_argument("--only", default="paderborn-baseline,cwru-class-imbalance")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    tasks = [task for task in load_tasks(args.tasks) if task.task_id in set(args.only.split(","))]
    rows = []
    for task in tasks:
        for condition, env_extra in CONDITIONS.items():
            for repeat in range(args.repeats):
                output = args.output / f"{task.task_id}-{condition}-r{repeat}"
                print(f"running {task.task_id} / {condition} / r{repeat} ...", flush=True)
                row = run_condition(task, condition, env_extra, output)
                row["repeat"] = repeat
                rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ablation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import statistics

    print()
    header = (
        f"{'task':22s} {'condition':13s} {'n':>2s} {'score':>7s} {'ungrounded':>12s} "
        f"{'unstable':>10s} {'runs_with_findings':>18s}"
    )
    print(header)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["condition"]), []).append(row)
    aggregate = []
    for (task_id, condition), items in grouped.items():
        scores = [item["score"] for item in items]
        ungrounded = [item.get("ungrounded_hypotheses", 0) for item in items]
        unstable = [item.get("unstable_experiments", 0) for item in items]
        findings = [item.get("review_findings", 0) for item in items]
        entry = {
            "task": task_id,
            "condition": condition,
            "runs": len(items),
            "score_mean": round(statistics.mean(scores), 4),
            "score_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "ungrounded_total": sum(ungrounded),
            "ungrounded_mean": round(statistics.mean(ungrounded), 3),
            "unstable_total": sum(unstable),
            "runs_with_findings": sum(1 for value in findings if value > 0),
        }
        aggregate.append(entry)
        print(
            f"{task_id:22s} {condition:13s} {entry['runs']:2d} "
            f"{entry['score_mean']:7.2f} {entry['ungrounded_total']:12d} "
            f"{entry['unstable_total']:10d} {entry['runs_with_findings']:18d}"
        )

    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
