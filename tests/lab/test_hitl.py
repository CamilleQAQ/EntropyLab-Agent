from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.lab import FakeLLM, LabContext, run_lab


def _ctx(tiny_dataset, tmp_path: Path) -> LabContext:
    spec, root = tiny_dataset
    return LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab", budget_seconds=600.0)


def _action(**payload) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_interactive_run_pauses_before_concluding(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    replies = [
        _action(
            action="run_experiment",
            args={"feature": "MDE", "classifier": "SVM-RBF", "seeds": 3},
            hypothesis="预计 MDE 与 SVM 稳定，用 3 个划分验证。",
            reason="baseline",
        ),
        _action(action="finish", reason="done"),
    ]
    prompts: list[dict] = []

    def ask(payload: dict) -> bool:
        prompts.append(payload)
        return True

    summary = run_lab(
        ctx, goal="maximise macro-F1", max_rounds=3, llm=FakeLLM(replies), interactive=True, ask=ask
    )
    assert prompts, "the graph must pause before finish"
    assert prompts[0]["action"] == "finish"
    assert summary["status"] == "done"


def test_interactive_rejection_stops_the_run(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    replies = [_action(action="finish", reason="done")]
    summary = run_lab(
        ctx,
        goal="maximise macro-F1",
        max_rounds=3,
        llm=FakeLLM(replies),
        interactive=True,
        ask=lambda payload: False,
    )
    assert summary["status"] in {"rejected", "done"}
    assert (ctx.workspace / "report.md").is_file()


def test_non_interactive_runs_do_not_pause(tiny_dataset, tmp_path: Path) -> None:
    ctx = _ctx(tiny_dataset, tmp_path)
    replies = [_action(action="finish", reason="done")]
    summary = run_lab(ctx, goal="x", max_rounds=2, llm=FakeLLM(replies))
    assert summary["status"] == "done"
