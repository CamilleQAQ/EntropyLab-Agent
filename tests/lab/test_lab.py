from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.lab import FakeLLM, LabContext, Ledger, run_lab
from entropy_agent.lab.reviewer import review_experiment
from entropy_agent.lab.tools import AnalyzeErrorsArgs, RunExperimentArgs, analyze_errors, run_experiment


def _ctx(tiny_dataset, tmp_path: Path) -> LabContext:
    spec, root = tiny_dataset
    return LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab", budget_seconds=600.0)


def _action(**payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_tool_run_experiment_returns_seed_statistics(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    result = run_experiment(ctx, RunExperimentArgs(feature="MDE", classifier="SVM-RBF", seeds=3))
    assert result["seeds"] == 3
    assert len(result["per_seed"]) == 3
    assert 0.0 <= result["macro_f1_mean"] <= 1.0
    assert result["macro_f1_std"] >= 0.0


def test_tool_analyze_errors_returns_confusion(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    result = analyze_errors(ctx, AnalyzeErrorsArgs(feature="MDE", classifier="SVM-RBF", seed=7))
    assert len(result["confusion"]) == len(result["labels"])
    assert sum(sum(row) for row in result["confusion"]) > 0


def test_reviewer_rejects_single_split() -> None:
    verdict = review_experiment(
        {"tool": "run_experiment", "seeds": 1, "macro_f1_mean": 0.9, "macro_f1_std": 0.0},
        baseline=None,
        claim="",
    )
    assert verdict["verdict"] == "revise"
    assert any(f["code"] == "TOO_FEW_SEEDS" for f in verdict["findings"])


def test_reviewer_rejects_a_claim_within_noise() -> None:
    verdict = review_experiment(
        {"tool": "run_experiment", "seeds": 5, "macro_f1_mean": 0.905, "macro_f1_std": 0.01},
        baseline={"macro_f1_mean": 0.900, "macro_f1_std": 0.01},
        claim="the new feature improves accuracy",
    )
    assert any(f["code"] == "CLAIM_WITHIN_NOISE" for f in verdict["findings"])


def test_lab_loop_profiles_then_retries_with_enough_seeds(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    replies = [
        _action(action="profile_dataset", reason="know the data"),
        _action(
            action="run_experiment",
            args={"feature": "MDE", "classifier": "SVM-RBF", "seeds": 1},
            hypothesis="MDE is enough",
            reason="cheap first",
        ),
        _action(
            action="run_experiment",
            args={"feature": "MDE", "classifier": "SVM-RBF", "seeds": 5},
            hypothesis="MDE is enough",
            reason="reviewer asked for more splits",
        ),
        _action(action="finish", reason="budget spent"),
    ]
    summary = run_lab(ctx, goal="maximise macro-F1", max_rounds=5, llm=FakeLLM(replies))

    assert summary["status"] == "done"
    assert len(summary["experiments"]) >= 3
    codes = {finding["code"] for review in summary["reviews"] for finding in review["findings"]}
    assert "TOO_FEW_SEEDS" in codes
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "研究日志" in report
    assert "macro-F1" in report


def test_ledger_persists_entries(tmp_path: Path) -> None:
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        ledger.record_hypothesis(1, "inner ring needs envelope features", "recall 0.42")
        ledger.record_experiment(1, "run_experiment", {"feature": "MDE"})
        ledger.record_review(1, "revise", [{"code": "TOO_FEW_SEEDS"}])
        assert ledger.hypotheses()[0]["text"].startswith("inner ring")
        assert ledger.experiments()[0]["tool"] == "run_experiment"
        assert ledger.reviews()[0]["verdict"] == "revise"
