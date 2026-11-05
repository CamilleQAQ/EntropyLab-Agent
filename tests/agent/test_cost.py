from __future__ import annotations

from pathlib import Path

import pytest

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, run_agent
from entropy_agent.agent import ExperimentMemory
from entropy_agent.agent.cost import BudgetLedger, CostModel


def _pool():
    return build_pool(
        methods=(
            MethodSpec("TSMEFuDE", {"m": 2, "nc": 4, "tau": 1, "kmax": 10}),
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("FUSION_TIME", {}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
        ),
    )


def test_price_scales_with_windows() -> None:
    model = CostModel()
    assert model.estimate("FUSION_TIME", 100) < model.estimate("FUSION_TIME", 1000)
    assert model.estimate("TSMEFuDE", 300) > 100 * model.estimate("FUSION_TIME", 300)


def test_observation_updates_the_price() -> None:
    model = CostModel()
    before = model.estimate("TSMEFuDE", 300)
    model.observe("TSMEFuDE", 300, 346.1)
    after = model.estimate("TSMEFuDE", 300)
    assert after > before
    assert 156.0 < after < 346.1
    model.observe("TSMEFuDE", 300, 0.0)  # cached; must be ignored
    assert model.estimate("TSMEFuDE", 300) == after


def test_cost_model_can_be_seeded_from_memory(tmp_path: Path) -> None:
    with ExperimentMemory(tmp_path / "memory.sqlite") as memory:
        memory.start_run("seed-run", "TINY", {}, {})
        memory.record_trial(
            run_id="seed-run", arm="agent", seed=1, trial_index=0, round_index=0,
            candidate={
                "candidate_id": "c0", "method": "MFE", "family": "fuzzy",
                "method_parameters": {}, "classifier": "SVM-RBF",
                "classifier_backend": "scikit-learn", "classifier_parameters": {},
            },
            feature_key="k", cache_hit=False, extracted_records=200, status="OK",
            failure=None, val_accuracy=0.5, val_macro_f1=0.5,
            val_accuracy_std=0.0, val_macro_f1_std=0.0, reason="seed",
            extract_seconds=40.0, fit_seconds=0.1,
        )
    model = CostModel.from_memory(tmp_path / "memory.sqlite")
    assert model.per_window["MFE"] == pytest.approx(0.2, rel=1e-6)


def test_ledger_charges_and_refuses() -> None:
    ledger = BudgetLedger(10.0)
    assert ledger.can_afford(9.0)
    ledger.charge("MDE", 9.0)
    assert ledger.remaining == pytest.approx(1.0)
    assert not ledger.can_afford(2.0)
    ledger.skip("MFE", 120.0, "does not fit the remaining budget")
    assert ledger.skipped_units[0]["method"] == "MFE"


def test_seconds_budget_pays_only_what_it_can_afford(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "seconds"
    summary = run_agent(
        spec,
        root,
        output,
        budget_seconds=1.0,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent",),
        pool=_pool(),
        run_test=False,
        run_id="tiny-seconds",
    )

    budget = summary["budget"]
    assert budget["mode"] == "seconds"
    assert budget["budget_seconds"] == 1.0
    ledgers = summary["ledgers"]["7"]["agent"]
    assert ledgers["spent_seconds"] <= 1.0 + 1e-9
    # TSMEFuDE is far too expensive for one second; the cheap units are paid first.
    assert "TSMEFuDE" not in ledgers["paid_units"]
    assert "FUSION_TIME" in ledgers["paid_units"]
    assert (output / "cost_model.json").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "秒预算账本" in report


def test_seconds_budget_pays_everything_when_generous(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "seconds-large"
    summary = run_agent(
        spec,
        root,
        output,
        budget_seconds=1000.0,
        seeds=(7,),
        validation_repeats=1,
        arms=("agent",),
        pool=_pool(),
        run_test=False,
        run_id="tiny-seconds-large",
    )
    ledgers = summary["ledgers"]["7"]["agent"]
    assert set(ledgers["paid_units"]) == {"TSMEFuDE", "MDE", "FUSION_TIME"}


def test_units_and_seconds_are_mutually_exclusive(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    with pytest.raises(ValueError, match="either units or budget_seconds"):
        run_agent(
            spec,
            root,
            tmp_path / "both",
            units=1,
            budget_seconds=5.0,
            seeds=(7,),
            arms=("agent",),
            pool=_pool(),
            run_test=False,
            run_id="tiny-both",
        )
