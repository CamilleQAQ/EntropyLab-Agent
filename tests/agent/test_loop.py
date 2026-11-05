from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.agent import (
    ClassifierSpec,
    ExperimentMemory,
    MethodSpec,
    build_pool,
    run_agent,
)


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
        seeds=(7,),
        validation_repeats=2,
        arms=("agent", "exhaustive"),
        pool=_pool(),
        run_id="tiny-run",
    )

    assert summary["aggregate"]["agent"]["per_seed"]["7"]["best_val_macro_f1"] is not None
    assert summary["aggregate"]["exhaustive"]["per_seed"]["7"]["best_val_macro_f1"] is not None
    assert summary["champion"] is not None
    assert summary["champion_seed"] == 7
    assert summary["test"] is not None
    assert summary["test"]["test"]["multiclass_accuracy"] is not None
    assert summary["champion_validation"]["macro_f1_std"] is not None

    for name in ("report.md", "trials.csv", "summary.json", "memory.sqlite"):
        assert (output / name).is_file()
    assert (output / "champion_model" / "config.json").is_file()

    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-run")
        assert len(trials) == 8
        stored = memory.run("tiny-run")
        assert stored is not None
        assert stored["champion"] is not None
        assert stored["test"] is not None
        assert all(trial["reason"] for trial in trials)


def test_features_are_shared_across_seeds_and_arms(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "shared-cache"
    run_agent(
        spec,
        root,
        output,
        rounds=1,
        per_round=2,
        seeds=(11, 12),
        validation_repeats=1,
        arms=("agent", "random"),
        pool=_pool(),
        run_test=False,
        run_id="tiny-shared",
    )

    # Two entropy methods, two seeds, two arms: still one cache entry per method.
    assert len(list((output / "feature_cache").glob("*.npz"))) == 2

    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-shared")
        assert {trial["seed"] for trial in trials} == {11, 12}
        # Every seed except the first reuses rows that already exist.
        warm = [trial for trial in trials if trial["seed"] == 12]
        assert all(trial["cache_hit"] for trial in warm)


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
        seeds=(7,),
        validation_repeats=1,
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
        seeds=(7, 8),
        validation_repeats=1,
        arms=("agent",),
        pool=_pool(),
        run_test=False,
        run_id="tiny-summary",
    )

    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert stored["run_id"] == summary["run_id"] == "tiny-summary"
    assert stored["aggregate"] == summary["aggregate"]
    assert set(stored["aggregate"]["agent"]["per_seed"]) == {"7", "8"}


# --- v1.2: the budget is a feature extraction, not a trial -------------------


def test_one_unit_buys_every_classifier_of_that_method(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "unit-mode"
    summary = run_agent(
        spec,
        root,
        output,
        units=1,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent",),
        pool=_pool(),  # two methods x two classifiers = two units
        run_test=False,
        run_id="tiny-units",
    )

    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-units")
    # One paid unit must evaluate both classifiers of exactly one method.
    assert len(trials) == 2
    assert len({trial["method"] for trial in trials}) == 1
    assert {trial["classifier"] for trial in trials} == {"SVM-RBF", "EXTRA-TREES"}
    assert summary["budget"]["units"] == 1
    assert summary["budget"]["mode"] == "units"


def test_unit_agent_starts_from_the_anchor_method(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "unit-anchor"
    pool = build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("TSMEFuDE", {"m": 2, "nc": 4, "tau": 1, "kmax": 10}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("BT-LSTSVM", "torch", {"c1": 1.0, "c2": 1.0, "bandwidth": 1.0, "epsilon": 1e-4}),
        ),
    )
    run_agent(
        spec,
        root,
        output,
        units=1,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent",),
        pool=pool,
        run_test=False,
        run_id="tiny-anchor-unit",
    )

    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-anchor-unit")
    assert {trial["method"] for trial in trials} == {"TSMEFuDE"}


def test_unit_exhaustive_covers_every_method(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "unit-exhaustive"
    run_agent(
        spec,
        root,
        output,
        units=1,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent", "exhaustive"),
        pool=_pool(),
        run_test=False,
        run_id="tiny-unit-exhaustive",
    )

    with ExperimentMemory(output / "memory.sqlite") as memory:
        agent = memory.trials("tiny-unit-exhaustive", arm="agent")
        exhaustive = memory.trials("tiny-unit-exhaustive", arm="exhaustive")
    assert len({trial["method"] for trial in agent}) == 1
    assert len({trial["method"] for trial in exhaustive}) == 2


def test_units_above_the_pool_size_is_rejected(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    try:
        run_agent(
            spec,
            root,
            tmp_path / "too-many-units",
            units=99,
            seeds=(7,),
            arms=("agent",),
            pool=_pool(),
            run_test=False,
            run_id="tiny-too-many",
        )
    except ValueError as error:
        assert "units" in str(error)
    else:  # pragma: no cover
        raise AssertionError("units above the pool size must be rejected")
