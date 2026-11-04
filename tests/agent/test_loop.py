from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.agent import ClassifierSpec, ExperimentMemory, MethodSpec, build_pool, run_agent


def _pool(methods_extra: tuple[MethodSpec, ...] = ()):
    return build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10}),
        )
        + methods_extra,
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
        ),
    )


def test_agent_runs_rounds_and_opens_test_once(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "run"
    summary = run_agent(
        spec,
        root,
        output,
        rounds=2,
        per_round=2,
        arms=("agent", "exhaustive"),
        pool=_pool(),
        run_id="tiny-run",
    )

    assert summary["arms"]["agent"]["trials"] == 4
    assert summary["arms"]["exhaustive"]["trials"] == 4
    assert summary["champion"] is not None
    assert summary["test"] is not None
    assert summary["test"]["test"]["multiclass_accuracy"] is not None

    for name in ("report.md", "trials.csv", "summary.json", "memory.sqlite"):
        assert (output / name).is_file()
    assert (output / "champion_model" / "config.json").is_file()

    with ExperimentMemory(output / "memory.sqlite") as memory:
        assert len(memory.trials("tiny-run")) == 8
        stored = memory.run("tiny-run")
        assert stored is not None
        assert stored["champion"] is not None
        assert stored["test"] is not None


def test_search_does_not_touch_the_test_partition(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "no-test"
    summary = run_agent(
        spec,
        root,
        output,
        rounds=1,
        per_round=2,
        arms=("agent",),
        pool=_pool(),
        run_test=False,
        run_id="tiny-no-test",
    )

    assert summary["test"] is None
    assert not (output / "champion_model").exists()
    cache_files = list((output / "feature_cache").glob("*.npz"))
    # One entry per distinct entropy method over the development records only.
    assert len(cache_files) == 2


def test_failed_trial_is_recorded_and_does_not_abort_the_run(
    tiny_dataset, tmp_path: Path
) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "failure"
    pool = build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("NOT-A-METHOD", {"m": 2}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
        ),
    )
    summary = run_agent(
        spec,
        root,
        output,
        rounds=1,
        per_round=2,
        arms=("agent",),
        pool=pool,
        run_test=False,
        run_id="tiny-failure",
    )

    assert summary["champion"] is not None  # the healthy candidate still wins
    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-failure")
    assert any(trial["status"] == "FAILED" for trial in trials)


def test_summary_json_matches_returned_dict(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "summary"
    summary = run_agent(
        spec,
        root,
        output,
        rounds=1,
        per_round=2,
        arms=("agent",),
        pool=_pool(),
        run_test=False,
        run_id="tiny-summary",
    )

    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert stored["run_id"] == summary["run_id"] == "tiny-summary"
    assert stored["arms"] == summary["arms"]
