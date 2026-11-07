from __future__ import annotations

from pathlib import Path

from entropy_agent.lab.eval import Task, load_tasks, score


def _summary(tmp_path: Path, **overrides) -> dict:
    report = tmp_path / "report.md"
    report.write_text("# 研究日志\n\n内容", encoding="utf-8")
    base = {
        "status": "done",
        "report": str(report),
        "experiments": [
            {"tool": "run_experiment", "feature": "FUSION_TIME", "seeds": 5, "macro_f1_mean": 0.9}
        ],
        "hypotheses": [{"round": 1, "text": "观察到 0.9，预计换分类器会更好", "evidence": "reason"}],
    }
    base.update(overrides)
    return base


def test_a_good_run_scores_full(tmp_path: Path) -> None:
    result = score({"min_experiments": 1, "min_seeds": 3}, _summary(tmp_path))
    assert result["score"] == 1.0
    assert result["failed"] == []


def test_missing_experiment_fails(tmp_path: Path) -> None:
    result = score({"min_experiments": 1}, _summary(tmp_path, experiments=[]))
    assert "has_experiment" in result["failed"]
    assert result["score"] < 1.0


def test_too_few_seeds_fails(tmp_path: Path) -> None:
    experiments = [{"tool": "run_experiment", "feature": "MDE", "seeds": 1}]
    result = score({"min_seeds": 3}, _summary(tmp_path, experiments=experiments))
    assert "enough_seeds" in result["failed"]


def test_repeated_calls_fail(tmp_path: Path) -> None:
    experiments = [
        {"tool": "run_experiment", "feature": "MDE", "seeds": 5},
        {"tool": "run_experiment", "error": "REPEATED_ACTION: already ran"},
    ]
    result = score({}, _summary(tmp_path, experiments=experiments))
    assert "no_repeated_calls" in result["failed"]
    assert "no_tool_failures" in result["failed"]


def test_ungrounded_hypotheses_fail(tmp_path: Path) -> None:
    hypotheses = [{"round": 1, "text": "x", "evidence": "未通过校验: 引用了没有的数字"}]
    result = score({}, _summary(tmp_path, hypotheses=hypotheses))
    assert "hypotheses_grounded" in result["failed"]


def test_missing_report_fails(tmp_path: Path) -> None:
    result = score({}, _summary(tmp_path, report=str(tmp_path / "nope.md")))
    assert "report_written" in result["failed"]


def test_task_loader_reads_the_config() -> None:
    tasks = load_tasks(Path("configs/lab_tasks.json"))
    assert len(tasks) >= 3
    assert all(isinstance(task, Task) for task in tasks)
    assert tasks[0].task_id


def test_ledger_shaped_rows_are_scored(tmp_path: Path) -> None:
    experiments = [
        {"round": 1, "tool": "run_experiment", "payload": {"tool": "run_experiment", "seeds": 5}}
    ]
    result = score({"min_seeds": 3}, _summary(tmp_path, experiments=experiments))
    assert result["score"] == 1.0, result["failed"]
