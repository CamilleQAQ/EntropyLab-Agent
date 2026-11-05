from __future__ import annotations

from pathlib import Path

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, run_agent
from entropy_agent.agent import ExperimentMemory
from entropy_agent.agent.feasibility import check, required_window


def test_default_pool_offers_entropy_and_fusion_families() -> None:
    pool = build_pool()
    methods = {candidate.method.method for candidate in pool}
    assert {"TSMEFuDE", "MDE"} <= methods
    assert {"FUSION_TIME", "FUSION_TIME_SPEC", "FUSION_TIME_ENVELOPE",
            "FUSION_TIME_ENVELOPE_ENTROPY", "FUSION_ALL"} <= methods
    families = {candidate.family for candidate in pool}
    assert "time_statistics" in families
    assert "time_spectral" in families
    assert "unknown" not in families


def test_fusion_units_are_counted_separately_from_entropy_units() -> None:
    pool = build_pool()
    units = {
        (candidate.method.method, tuple(sorted(candidate.method.parameters.items())))
        for candidate in pool
    }
    # 8 entropy methods + 5 fusion recipes, each one unit.
    assert len(units) == 13


def test_fusion_feasibility_only_needs_a_minimum_window() -> None:
    assert required_window("FUSION_TIME", {}) == 16
    assert check("FUSION_TIME", {}, 2000).feasible is True
    assert check("FUSION_ALL", {}, 16).feasible is True
    below = check("FUSION_TIME_SPEC", {}, 8)
    assert below.feasible is False
    assert "16" in below.reason


def test_agent_evaluates_a_fusion_unit(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "fusion-unit"
    pool = build_pool(
        methods=(
            MethodSpec("FUSION_TIME_SPEC", {}),
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
        ),
        classifiers=(
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
        ),
    )
    summary = run_agent(
        spec,
        root,
        output,
        units=1,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent",),
        pool=pool,
        run_test=False,
        run_id="tiny-fusion",
    )

    with ExperimentMemory(output / "memory.sqlite") as memory:
        trials = memory.trials("tiny-fusion")
    assert len(trials) == 1
    assert trials[0]["method"] == "FUSION_TIME_SPEC"
    assert trials[0]["status"] == "OK"
    assert summary["aggregate"]["agent"]["per_seed"]["7"]["best_val_macro_f1"] is not None
