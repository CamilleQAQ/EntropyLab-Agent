"""Task-level evaluation for the research lab.

Each task is a dataset + goal + expectations.  The score is not "did it find a
good model" but "did it run the research process properly": experiments actually
ran, enough splits, no repeated calls, grounded hypotheses, a report with limits.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_TOOLS = {"run_experiment", "run_custom_experiment"}


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    goal: str
    max_rounds: int = 4
    expect: dict | None = None


def load_tasks(path: Path) -> list[Task]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Task(
            task_id=entry["id"],
            dataset=entry["dataset"],
            goal=entry["goal"],
            max_rounds=int(entry.get("max_rounds", 4)),
            expect=entry.get("expect", {}),
        )
        for entry in payload["tasks"]
    ]


def _flat(item: dict) -> dict:
    """Ledger rows nest the tool result under `payload`; accept both shapes."""
    payload = item.get("payload")
    return payload if isinstance(payload, dict) else item


def grid_ranking(grid_path: pathlib.Path, dataset: str) -> list[tuple[str, str]]:
    """True candidate ranking for a dataset, from the collected experiment grid."""
    import csv

    import numpy as np

    scores: dict[tuple[str, str], list[float]] = {}
    with pathlib.Path(grid_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset:
                continue
            scores.setdefault((row["feature"], row["classifier"]), []).append(float(row["macro_f1"]))
    return [
        candidate
        for candidate, _ in sorted(scores.items(), key=lambda item: -float(np.mean(item[1])))
    ]


def _recommended(experiments: list[dict]) -> tuple[str, str] | None:
    """The candidate the agent would recommend: best validation score it ran."""
    scored = [
        (float(item["macro_f1_mean"]), (item["feature"], item["classifier"]))
        for item in experiments
        if item.get("feature") and item.get("classifier") and item.get("macro_f1_mean") is not None
    ]
    return max(scored)[1] if scored else None


def score(expect: dict | None, summary: dict, *, truth: list[tuple[str, str]] | None = None) -> dict:
    expect = expect or {}
    checks: dict[str, bool] = {}
    rows = [_flat(item) for item in summary.get("experiments", [])]

    checks["finished"] = summary.get("status") == "done"

    experiments = [item for item in rows if item.get("tool") in EXPERIMENT_TOOLS]
    checks["has_experiment"] = len(experiments) >= int(expect.get("min_experiments", 1))

    min_seeds = int(expect.get("min_seeds", 3))
    checks["enough_seeds"] = bool(experiments) and all(
        int(item.get("seeds") or 0) >= min_seeds for item in experiments
    )

    failures = [item for item in rows if item.get("error")]
    checks["no_tool_failures"] = len(failures) <= int(expect.get("max_tool_failures", 0))
    checks["no_repeated_calls"] = not any(
        "REPEATED_ACTION" in str(item.get("error", "")) for item in rows
    )

    report = Path(summary.get("report", ""))
    checks["report_written"] = (
        report.is_file() and "研究日志" in report.read_text(encoding="utf-8")
    )

    hypotheses = summary.get("hypotheses", [])
    checks["has_hypothesis"] = len(hypotheses) >= int(expect.get("min_hypotheses", 1))
    checks["hypotheses_grounded"] = any(
        "未通过校验" not in (row.get("evidence") or "") for row in hypotheses
    )

    top_k = int(expect.get("top_k", 0) or 0)
    if top_k and truth:
        recommended = _recommended(experiments)
        if recommended is None:
            checks["conclusion_in_top_k"] = False
        else:
            rank = truth.index(recommended) if recommended in truth else len(truth)
            checks["conclusion_in_top_k"] = rank < top_k

    report_text = ""
    if checks.get("report_written"):
        report_text = report.read_text(encoding="utf-8")

    if expect.get("require_per_class_recall"):
        checks["reports_per_class_recall"] = (
            "recall" in report_text.lower() or "召回" in report_text
        )
    if expect.get("require_control"):
        checks["runs_a_control"] = any(
            keyword in report_text
            for keyword in ("control", "对照", "下采样", "downsampl", "baseline comparison")
        )

    passed = sum(1 for value in checks.values() if value)
    total = len(checks)
    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 4),
        "failed": [name for name, value in checks.items() if not value],
    }
