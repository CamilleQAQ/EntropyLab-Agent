"""Phase 2 contract tests for the Paderborn baseline configuration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from entropy_agent.io.dataset import (
    DatasetSpec,
    load_dataset_records,
    load_dataset_spec,
    resolve_dataset_split,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paderborn_baseline.json"


def _spec() -> DatasetSpec:
    return load_dataset_spec(CONFIG)


def test_paderborn_config_loads_and_manifest_is_development_only() -> None:
    spec = _spec()
    assert spec.dataset_id == "PADERBORN_3CLASS_64KHZ_V2"
    assert spec.split.mode == "manifest"
    assert spec.split.expected_records == {"train": 640, "validation": 320, "test": 0}
    assert spec.window.size == 4096
    assert spec.window.stride == 4096
    assert spec.feature.method == "MDE"
    assert spec.classifier.algorithm == "SVM-RBF"
    assert "sample_stride" not in spec.feature.parameters

    records = load_dataset_records(spec, ROOT)
    assert len(records) == 960
    split_by_group = resolve_dataset_split(spec, records)
    assert Counter(split_by_group.values()) == {"train": 8, "validation": 4}
    assert Counter(
        split_by_group[record.group] for record in records
    ) == {"train": 640, "validation": 320}

    for record in records:
        assert record.signal_input.path.exists(), record.signal_input.path


def test_paderborn_config_does_not_include_final_holdout_groups() -> None:
    import csv

    final_groups: set[str] = set()
    with (ROOT / "manifests" / "paderborn" / "final_holdout.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            final_groups.add(row["group_id"])

    spec = _spec()
    records = load_dataset_records(spec, ROOT)
    assert final_groups
    assert not ({record.group for record in records} & final_groups)


def test_single_entropy_method_rejects_hidden_sample_stride() -> None:
    spec = _spec()
    payload = spec.model_dump(mode="json")
    payload["feature"]["parameters"]["sample_stride"] = 8192
    with pytest.raises(ValidationError, match="sample_stride"):
        DatasetSpec.model_validate(payload, strict=False)


def test_manifest_split_allows_zero_test_records() -> None:
    spec = _spec()
    payload = spec.model_dump(mode="json")
    payload["split"]["expected_records"] = {"train": 640, "validation": 320, "test": 0}
    restored = DatasetSpec.model_validate(payload, strict=False)
    assert restored.split.expected_records["test"] == 0


def test_fixed_split_rejects_negative_or_zero_train_validation() -> None:
    spec = _spec()
    for mutation in (
        {"train": 0, "validation": 320, "test": 0},
        {"train": 640, "validation": 0, "test": 0},
        {"train": 640, "validation": 320, "test": -1},
    ):
        payload = spec.model_dump(mode="json")
        payload["split"]["expected_records"] = mutation
        with pytest.raises(ValidationError):
            DatasetSpec.model_validate(payload, strict=False)
