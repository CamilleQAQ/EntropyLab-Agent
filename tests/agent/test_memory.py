from __future__ import annotations

from pathlib import Path

from entropy_agent.agent import ExperimentMemory


def _record(
    memory: ExperimentMemory,
    *,
    index: int,
    status: str = "OK",
    seed: int = 1,
    run_id: str = "run-1",
) -> None:
    memory.record_trial(
        run_id=run_id,
        arm="agent",
        seed=seed,
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
        extracted_records=0 if index else 18,
        status=status,
        failure=None if status == "OK" else "boom",
        val_accuracy=0.5 + index / 100,
        val_macro_f1=0.4 + index / 100,
        val_accuracy_std=0.01,
        val_macro_f1_std=0.02,
        reason="exploit best method",
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
    assert trials[0]["extracted_records"] == 18
    assert trials[0]["val_macro_f1_std"] == 0.02
    assert trials[0]["reason"] == "exploit best method"
    assert trials[1]["status"] == "FAILED"
    assert trials[1]["failure"] == "boom"
    assert abs(trials[1]["val_macro_f1"] - 0.41) < 1e-9


def test_runs_are_separated_by_seed(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-2", "TINY", {}, {})
        _record(memory, index=0, seed=1, run_id="run-2")
        _record(memory, index=1, seed=1, run_id="run-2")
        _record(memory, index=0, seed=2, run_id="run-2")

        assert len(memory.trials("run-2", seed=1)) == 2
        assert len(memory.trials("run-2", seed=2)) == 1
        assert len(memory.trials("run-2")) == 3


def test_run_records_champion_and_test(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-3", "TINY", {}, {"rounds": 1})
        _record(memory, index=0)
        memory.finish_run("run-3", {"candidate_id": "c0"}, {"test": {"accuracy": 0.75}})
        stored = memory.run("run-3")

    assert stored is not None
    assert stored["champion"] == {"candidate_id": "c0"}
    assert stored["test"]["test"]["accuracy"] == 0.75
    assert stored["finished_at"] is not None


def test_starting_a_run_twice_clears_previous_trials(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("run-4", "TINY", {}, {})
        _record(memory, index=0)
        memory.start_run("run-4", "TINY", {}, {})
        assert memory.trials("run-4") == []


def test_v1_database_is_migrated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trials (
            run_id TEXT NOT NULL, arm TEXT NOT NULL, trial_index INTEGER NOT NULL,
            round_index INTEGER NOT NULL, candidate_id TEXT NOT NULL, method TEXT NOT NULL,
            family TEXT NOT NULL, method_parameters TEXT NOT NULL, classifier TEXT NOT NULL,
            classifier_backend TEXT NOT NULL, classifier_parameters TEXT NOT NULL,
            feature_key TEXT NOT NULL, cache_hit INTEGER NOT NULL, status TEXT NOT NULL,
            failure TEXT, val_accuracy REAL, val_macro_f1 REAL, extract_seconds REAL NOT NULL,
            fit_seconds REAL NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, arm, trial_index)
        );
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
            profile_json TEXT NOT NULL, budget_json TEXT NOT NULL, started_at TEXT NOT NULL,
            finished_at TEXT, champion_json TEXT, test_json TEXT);
        """
    )
    connection.commit()
    connection.close()

    with ExperimentMemory(path) as memory:
        memory.start_run("run-5", "TINY", {}, {})
        _record(memory, index=0, run_id="run-5")
        trials = memory.trials("run-5")

    assert trials[0]["seed"] == 1
    assert trials[0]["reason"] == "exploit best method"
