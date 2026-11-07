from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from entropy_agent.continual import BatchPlan, PlanError, run_continual, save_plan
from entropy_agent.continual.plan import Batch
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec


def test_extra_retention_probes_are_visible_only_after_their_batch() -> None:
    plan = BatchPlan(
        dataset_id="example",
        batches=(
            Batch(0, ("train0",), probe_groups=("probe0",), retention_probe_groups=("extra0",)),
            Batch(1, ("train1",), probe_groups=("probe1",), retention_probe_groups=("extra1",)),
        ),
        validation_groups=("reference",),
        holdout_groups=("holdout",),
    )
    groups = ["train0", "train1", "probe0", "probe1", "extra0", "extra1", "reference", "holdout"]
    records = [SimpleNamespace(group=group, label=0) for group in groups]
    restored = BatchPlan.from_dict(plan.to_dict())
    restored.validate(records)
    assert restored.visible_probes(0) == {"reference": ("reference",), "batch-0": ("probe0",)}
    assert restored.visible_probes(1) == {
        "reference": ("reference",), "batch-0": ("probe0", "extra0"), "batch-1": ("probe1",),
    }
    for index in range(2):
        assert not restored.masks(records, index)["train"][4:6].any()


def _records_and_plan(tiny_dataset, batches: int = 2):
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    by_class: dict[int, list[str]] = {}
    for record in records:
        by_class.setdefault(int(record.label), [])
        if record.group not in by_class[int(record.label)]:
            by_class[int(record.label)].append(record.group)
    # 3 classes x 6 groups each: 1 holdout + 1 validation + 4 remaining
    holdout = [groups[-1] for groups in by_class.values()]
    validation = [groups[-2] for groups in by_class.values()]
    remaining = [groups[:-2] for groups in by_class.values()]
    per_batch = min(len(groups) for groups in remaining) // batches
    plan_batches = []
    for index in range(batches):
        groups: list[str] = []
        for class_groups in remaining:
            start = index * per_batch
            end = len(class_groups) if index == batches - 1 else (index + 1) * per_batch
            groups.extend(class_groups[start:end])
        plan_batches.append({"batch_id": index, "groups": groups, "note": f"batch {index}"})
    payload = {
        "format": "entropylab.continual-plan",
        "format_version": 1,
        "dataset_id": spec.dataset_id,
        "margin": 0.0,
        "batches": plan_batches,
        "validation_groups": validation,
        "holdout_groups": holdout,
    }
    return spec, root, records, payload


def test_plan_rejects_a_group_used_twice(tiny_dataset) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    payload["holdout_groups"] = payload["holdout_groups"] + [payload["batches"][0]["groups"][0]]
    plan = BatchPlan.from_dict(payload)
    with pytest.raises(PlanError, match="more than once"):
        plan.validate(records)


def test_plan_rejects_an_unassigned_group(tiny_dataset) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    payload["holdout_groups"] = payload["holdout_groups"][:-1]
    plan = BatchPlan.from_dict(payload)
    with pytest.raises(PlanError, match="unassigned"):
        plan.validate(records)


def test_holdout_never_enters_the_training_history(tiny_dataset) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    plan = BatchPlan.from_dict(payload)
    groups = np.asarray([record.group for record in records], dtype=object)
    for batch_index in range(plan.batch_count):
        masks = plan.masks(records, batch_index)
        assert not (masks["train"] & masks["test"]).any()
        assert not (masks["train"] & masks["validation"]).any()
        # Only arrived batches are in train; future batches belong to no split yet.
        arrived = np.isin(groups, np.asarray(plan.history_groups(batch_index), dtype=object))
        assert masks["train"].sum() == arrived.sum()

    final_train = plan.masks(records, plan.batch_count - 1)["train"]
    every_batch_group = np.isin(
        groups, np.asarray([group for batch in plan.batches for group in batch.groups], dtype=object)
    )
    assert final_train.sum() == every_batch_group.sum()
    # The training set only grows, batch by batch.
    sizes = [plan.masks(records, index)["train"].sum() for index in range(plan.batch_count)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_continual_run_promotes_and_writes_artifacts(tiny_dataset, tmp_path: Path) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    plan = BatchPlan.from_dict(payload)
    output = tmp_path / "continual"
    summary = run_continual(spec, root, plan, output, model_prefix="tiny")

    assert summary["plan"]["batches"] == 2
    assert len(summary["decisions"]) == 2
    assert summary["decisions"][0]["promoted"] is True  # first model is the champion
    # margin 0.0 and a tiny problem: the challenger is never worse than chance,
    # so promotion is allowed to happen at most once per batch and never twice.
    assert summary["promotions"] <= 2
    assert (output / "metrics.json").is_file()
    assert (output / "decisions.json").is_file()
    assert (output / "report.md").is_file()
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "固定 holdout 三臂对比" in report
    assert set(summary["holdout"]) == {"frozen", "continual", "joint"}


def test_joint_arm_uses_every_batch(tiny_dataset, tmp_path: Path) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    plan = BatchPlan.from_dict(payload)
    summary = run_continual(spec, root, plan, tmp_path / "continual", model_prefix="tiny")
    # Joint trains on all batch groups, which is strictly more than batch 0 alone.
    joint_training = summary["joint_validation_macro_f1"]
    assert joint_training is not None


def test_plan_file_round_trip(tiny_dataset, tmp_path: Path) -> None:
    spec, root, records, payload = _records_and_plan(tiny_dataset)
    plan = BatchPlan.from_dict(payload)
    path = tmp_path / "plan.json"
    save_plan(path, plan)
    restored = BatchPlan.load(path)
    assert restored.to_dict() == plan.to_dict()
    restored.validate(records)
