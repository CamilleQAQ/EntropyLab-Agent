"""Task-level evaluation for the research lab.

Each task is a dataset + goal + expectations.  The score is not "did it find a
good model" but "did it run the research process properly": experiments actually
ran, enough splits, no repeated calls, grounded hypotheses, a report with limits.
"""

from __future__ import annotations

import json
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


def score(expect: dict | None, summary: dict) -> dict:
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

    passed = sum(1 for value in checks.values() if value)
    total = len(checks)
    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 4),
        "failed": [name for name, value in checks.items() if not value],
    }
