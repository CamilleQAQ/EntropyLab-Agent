from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.lab import FakeLLM, LabContext, run_lab
from entropy_agent.lab.grounding import problems

HISTORY = [
    {
        "tool": "run_experiment",
        "feature": "FUSION_TIME",
        "classifier": "EXTRA-TREES",
        "seeds": 5,
        "macro_f1_mean": 0.9611,
        "macro_f1_std": 0.007,
    }
]


def test_grounded_hypothesis_passes() -> None:
    text = "观察到 macro-F1 0.9611，预计换分类器能提升到 0.97，用 5 划分验证。"
    assert problems(text, HISTORY) == []


def test_fabricated_number_is_rejected() -> None:
    text = "观察到 macro-F1 0.99，预计换分类器会提升，用 5 划分验证。"
    found = problems(text, HISTORY)
    assert any("没有的数字" in item for item in found)


def test_missing_prediction_is_rejected() -> None:
    text = "观察到 macro-F1 0.9611。"
    found = problems(text, HISTORY)
    assert any("预测" in item for item in found)


def test_first_round_has_no_measurement_to_cite() -> None:
    text = "预计 ExtraTrees 在时间特征上更稳，用 5 划分验证。"
    assert problems(text, []) == []


def _action(**payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_planner_repairs_an_ungrounded_hypothesis(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    ctx = LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab", budget_seconds=600.0)
    replies = [
        # round 1: no history yet, hypothesis accepted
        _action(
            action="run_experiment",
            args={"feature": "MDE", "classifier": "SVM-RBF", "seeds": 5},
            hypothesis="预计 MDE 与 SVM 的组合能稳定分类，用 5 个划分验证。",
            reason="start with a cheap baseline",
        ),
        # round 2: fabricated number, must be repaired
        _action(
            action="run_experiment",
            args={"feature": "MPE", "classifier": "EXTRA-TREES", "seeds": 5},
            hypothesis="观察到 macro-F1 0.99，预计换特征能提升到 0.995。",
            reason="switch feature",
        ),
        # repair reply with a grounded number
        _action(
            action="run_experiment",
            args={"feature": "MPE", "classifier": "EXTRA-TREES", "seeds": 5},
            hypothesis="观察到上一轮 macro-F1 已由工具返回，预计换 MPE 后变化不超过 2 倍波动，用 5 个划分验证。",
            reason="switch feature",
        ),
        _action(action="finish", reason="done"),
    ]
    summary = run_lab(ctx, goal="maximise macro-F1", max_rounds=4, llm=FakeLLM(replies))
    evidence = " ".join(row["evidence"] or "" for row in summary["hypotheses"])
    assert "未通过校验" not in evidence, evidence
