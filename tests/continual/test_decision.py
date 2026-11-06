from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from entropy_agent.continual import BatchPlan, run_continual
from entropy_agent.continual.decision import decide
from entropy_agent.io.dataset import load_dataset_records


def test_margin_criterion_needs_a_strict_improvement() -> None:
    decision = decide(
        criterion="margin",
        champion_validation_macro_f1=0.95,
        challenger_validation_macro_f1=0.95,
        margin=0.005,
    )
    assert decision.promoted is False


def test_margin_criterion_cannot_fire_at_ceiling() -> None:
    decision = decide(
        criterion="margin",
        champion_validation_macro_f1=1.0,
        challenger_validation_macro_f1=1.0,
        margin=0.005,
    )
    assert decision.promoted is False
    assert "needs 1.0050" in decision.reason


def test_guardrail_promotes_at_ceiling_when_the_new_domain_improves() -> None:
    decision = decide(
        criterion="guardrail",
        champion_validation_macro_f1=1.0,
        challenger_validation_macro_f1=1.0,
        margin=0.005,
        retention_drop=0.0,
        gain_new=0.03,
        guard_tolerance=0.02,
    )
    assert decision.promoted is True
    assert decision.gain_new == pytest.approx(0.03)


def test_guardrail_blocks_a_update_that_hurts_retention() -> None:
    decision = decide(
        criterion="guardrail",
        champion_validation_macro_f1=0.9,
        challenger_validation_macro_f1=0.99,
        margin=0.005,
        retention_drop=0.05,
        gain_new=0.09,
        guard_tolerance=0.02,
    )
    assert decision.promoted is False
    assert "retention drop" in decision.reason


def test_guardrail_blocks_a_negligible_gain() -> None:
    decision = decide(
        criterion="guardrail",
        champion_validation_macro_f1=0.9,
        challenger_validation_macro_f1=0.9,
        margin=0.005,
        retention_drop=0.0,
        gain_new=0.001,
        min_gain=0.01,
    )
    assert decision.promoted is False
    assert "gain" in decision.reason


def _plan_with_probes(tiny_dataset) -> tuple[object, Path, BatchPlan]:
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    by_class: dict[int, list[str]] = {}
    for record in records:
        groups = by_class.setdefault(int(record.label), [])
        if record.group not in groups:
            groups.append(record.group)
    holdout = [groups[-1] for groups in by_class.values()]
    validation = [groups[-2] for groups in by_class.values()]
    remaining = [groups[:-2] for groups in by_class.values()]
    batch0 = [groups[0] for groups in remaining]
    batch1 = [groups[1] for groups in remaining]
    probe0 = [groups[2] for groups in remaining]
    probe1 = [groups[3] for groups in remaining]
    payload = {
        "format": "entropylab.continual-plan",
        "format_version": 1,
        "dataset_id": spec.dataset_id,
        "margin": 0.005,
        "batches": [
            {"batch_id": 0, "groups": batch0, "note": "round 0", "probe_groups": probe0},
            {"batch_id": 1, "groups": batch1, "note": "round 1", "probe_groups": probe1},
        ],
        "validation_groups": validation,
        "holdout_groups": holdout,
    }
    return spec, root, BatchPlan.from_dict(payload)


def test_guardrail_path_is_wired_end_to_end(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    summary = run_continual(
        spec,
        root,
        plan,
        tmp_path / "guardrail",
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=1.0,
        min_gain=-1.0,
    )
    decision = summary["decisions"][1]
    assert decision["promoted"] is True
    assert decision["gain_new"] is not None
    assert decision["retention_drop"] is not None
    assert set(decision["probe_scores"]["champion"]) == {"reference", "batch-0", "batch-1"}
    assert summary["criterion"] == "guardrail"


def test_margin_path_still_rejects_when_the_margin_is_unreachable(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    plan = BatchPlan(
        dataset_id=plan.dataset_id,
        batches=plan.batches,
        validation_groups=plan.validation_groups,
        holdout_groups=plan.holdout_groups,
        margin=0.9,
        notes=plan.notes,
    )
    summary = run_continual(
        spec, root, plan, tmp_path / "margin", model_prefix="tiny", criterion="margin"
    )
    assert summary["decisions"][1]["promoted"] is False
    assert summary["decisions"][1]["retention_drop"] is None


def test_probes_never_enter_training(tiny_dataset) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    records = load_dataset_records(spec, root)
    groups = np.asarray([record.group for record in records], dtype=object)
    for batch_index in range(plan.batch_count):
        masks = plan.masks(records, batch_index)
        for batch in plan.batches[: batch_index + 1]:
            if not batch.probe_groups:
                continue
            probe = np.isin(groups, np.asarray(batch.probe_groups, dtype=object))
            assert not (masks["train"] & probe).any()
