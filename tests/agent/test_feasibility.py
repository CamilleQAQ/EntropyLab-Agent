from __future__ import annotations

from entropy_agent.agent.feasibility import check, required_window


def test_required_window_follows_the_coarse_grain_and_pattern_rule() -> None:
    # scale x ((m - 1) * tau + 1)
    assert required_window("TSMEFuDE", {"m": 2, "tau": 1, "kmax": 20}) == 40
    assert required_window("MDE", {"m": 3, "tau": 2, "scale": 20}) == 100
    assert required_window("MPE", {"m": 3, "tau": 1, "scale": 20}) == 60
    assert required_window("MFE", {"m": 2, "r": 0.15, "n": 2, "tau": 1, "scale": 20}) == 40


def test_check_accepts_a_window_at_exactly_the_required_length() -> None:
    verdict = check("TSMEFuDE", {"m": 2, "tau": 1, "kmax": 20}, 40)
    assert verdict.feasible is True
    assert verdict.reason == ""

    below = check("TSMEFuDE", {"m": 2, "tau": 1, "kmax": 20}, 39)
    assert below.feasible is False
    assert "39" in below.reason and "40" in below.reason


def test_tau_and_scale_both_count(tiny_dataset, tmp_path) -> None:
    from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, run_agent
    from entropy_agent.agent import ExperimentMemory

    spec, root = tiny_dataset  # window = 200 samples
    pool = build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),  # 20 -> feasible
            MethodSpec("MPE", {"m": 3, "tau": 2, "scale": 40}),  # 200 -> feasible (exactly)
            MethodSpec("TSMEFuDE", {"m": 3, "tau": 2, "kmax": 40}),  # 200 -> feasible
            MethodSpec("MFuDE", {"m": 3, "tau": 2, "scale": 60}),  # 300 -> infeasible
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
        ),
    )
    output = tmp_path / "feasibility"
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
        run_id="tiny-feasibility",
    )

    assert summary["budget"]["pool_size"] == 3  # MFuDE dropped
    with ExperimentMemory(output / "memory.sqlite") as memory:
        run = memory.run("tiny-feasibility")
    skipped = run["budget"]["skipped_candidates"]
    assert [entry["method"] for entry in skipped] == ["MFuDE"]
    assert skipped[0]["required_window"] == 300
    assert "window 200" in skipped[0]["reason"]

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "可行性过滤" in report
    assert "MFuDE" in report and "300" in report


def test_every_candidate_infeasible_is_rejected(tiny_dataset, tmp_path) -> None:
    import pytest

    from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, run_agent

    spec, root = tiny_dataset
    pool = build_pool(
        methods=(MethodSpec("MDE", {"m": 3, "nc": 4, "tau": 2, "scale": 100}),),
        classifiers=(ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),),
    )
    with pytest.raises(ValueError, match="no candidate is feasible"):
        run_agent(
            spec,
            root,
            tmp_path / "empty-pool",
            units=1,
            seeds=(7,),
            arms=("agent",),
            pool=pool,
            run_test=False,
            run_id="tiny-empty",
        )
