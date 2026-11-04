from __future__ import annotations

from pathlib import Path

from entropy_agent.agent import ExperimentMemory


def _record(memory: ExperimentMemory, *, index: int, status: str = "OK") -> None:
    memory.record_trial(
        run_id="run-1",
        arm="agent",
        trial_index=index,
        round_index=index // 3,
        candidate={
            "candidate_id": f"c{index}",
            "method": "MDE",
            "family": "dispersion",
            "method_parameters": {"m": 2},
            "classifier": "SVM-RBF",
            "classifier_backend": "scikit-learn",
            "classifier_parameters": {"C": 1.0},
        },
        feature_key=f"key-{index}",
        cache_hit=index > 0,
        status=status,
        failure=None if status == "OK" else "boom",
        val_accuracy=0.5 + index / 100,
        val_macro_f1=0.4 + index / 100,
        extract_seconds=1.5,
        fit_seconds=0.01,
    )


def test_trials_round_trip_with_parameters(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-1", "TINY", {"class_count": 3}, {"rounds": 2})
        _record(memory, index=0)
        _record(memory, index=1, status="FAILED")

        trials = memory.trials("run-1")

    assert [trial["trial_index"] for trial in trials] == [0, 1]
    assert trials[0]["method_parameters"] == {"m": 2}
    assert trials[0]["cache_hit"] is False
    assert trials[1]["status"] == "FAILED"
    assert trials[1]["failure"] == "boom"
    assert abs(trials[1]["val_macro_f1"] - 0.41) < 1e-9


def test_run_records_champion_and_test(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-2", "TINY", {}, {"rounds": 1})
        _record(memory, index=0)
        memory.finish_run("run-2", {"candidate_id": "c0"}, {"test": {"accuracy": 0.75}})
        stored = memory.run("run-2")

    assert stored is not None
    assert stored["champion"] == {"candidate_id": "c0"}
    assert stored["test"]["test"]["accuracy"] == 0.75
    assert stored["finished_at"] is not None


def test_starting_a_run_twice_clears_previous_trials(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-3", "TINY", {}, {})
        _record(memory, index=0)
        memory.start_run("run-3", "TINY", {}, {})
        assert memory.trials("run-3") == []
