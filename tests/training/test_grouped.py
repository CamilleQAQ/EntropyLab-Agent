import numpy as np
import pytest

from entropy_agent.training.grouped import (
    evaluate_channel_candidates,
    stratified_group_split,
)


def test_stratified_group_split_is_disjoint_and_keeps_every_class() -> None:
    labels = np.repeat(np.arange(3), 10)
    groups = [f"class-{label}-group-{index}" for label in range(3) for index in range(10)]
    split = stratified_group_split(labels, groups, seed=17)
    assert set(split.values()) == {"train", "validation", "test"}
    for label in range(3):
        class_splits = {split[group] for group, value in zip(groups, labels) if value == label}
        assert class_splits == {"train", "validation", "test"}


def test_channel_selection_uses_voted_validation_records(monkeypatch) -> None:
    labels = np.tile(np.arange(3), 4)
    groups = [f"g-{index}" for index in range(len(labels))]
    split = {
        group: ("train" if index < 6 else "validation" if index < 9 else "test")
        for index, group in enumerate(groups)
    }
    features = np.zeros((len(labels), 3, 2, 20), dtype=float)
    features[:, :, :, 0] = labels[:, None, None]

    class FakeModel:
        calls = 0
        def __init__(self, **kwargs): pass
        def fit(self, x, y, class_order): return self
        def predict(self, x):
            channel = FakeModel.calls // 2
            is_validation = FakeModel.calls % 2 == 0
            FakeModel.calls += 1
            if channel == 0 and is_validation:
                return x[:, 0].astype(int)
            return np.zeros(len(x), dtype=int)

    monkeypatch.setattr(
        "entropy_agent.training.grouped.BTLSTSVM", FakeModel,
    )
    result = evaluate_channel_candidates(
        features, labels, groups, split,
        channel_names=("first", "second"), class_names=("a", "b", "c"),
    )
    assert result["selected_channel"] == "first"
    assert result["windows_per_record"] == 3


def test_optional_training_balance_does_not_change_validation_rows(monkeypatch) -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])
    groups = [f"g-{index}" for index in range(len(labels))]
    split = {
        group: ("train" if index in (0, 3, 4, 5) else "validation" if index in (1, 6) else "test")
        for index, group in enumerate(groups)
    }
    features = np.zeros((len(labels), 1, 1, 2), dtype=float)

    class FakeModel:
        fitted_rows = None
        def __init__(self, **kwargs): pass
        def fit(self, x, y, class_order):
            FakeModel.fitted_rows = len(y)
            return self
        def predict(self, x): return np.zeros(len(x), dtype=int)

    monkeypatch.setattr(
        "entropy_agent.training.grouped.BTLSTSVM", FakeModel,
    )
    result = evaluate_channel_candidates(
        features, labels, groups, split,
        channel_names=("only",), class_names=("healthy", "fault"),
        balance_training_classes=True,
    )
    assert FakeModel.fitted_rows == 6
    assert result["training_rows_before_balance"] == 4
    assert result["training_rows_after_balance"] == 6
    assert result["channels"][0]["validation"]["rows"] == 2


def test_grouped_pipeline_can_select_torch_backend() -> None:
    labels = np.tile(np.arange(2), 6)
    groups = [f"g-{index}" for index in range(len(labels))]
    split = {
        group: ("train" if index < 6 else "validation" if index < 10 else "test")
        for index, group in enumerate(groups)
    }
    features = np.zeros((len(labels), 1, 1, 2), dtype=float)
    features[:, 0, 0, 0] = np.where(labels == 0, -1.0, 1.0)
    features[:, 0, 0, 1] = np.linspace(-0.1, 0.1, len(labels))
    result = evaluate_channel_candidates(
        features, labels, groups, split,
        channel_names=("only",), class_names=("healthy", "fault"),
        classifier_backend="torch",
        model_parameters={"c1": 0.25, "c2": 0.25, "bandwidth": 4.0},
    )
    assert result["classifier"] == "BT-LSTSVM"
    assert result["classifier_backend"] == "torch"
    assert result["selected_channel_test"]["multiclass_accuracy"] == 1.0


def test_grouped_protocol_rejects_non_bt_backends() -> None:
    """The channel protocol is intentionally BT-LSTSVM-specific."""
    labels = np.tile(np.arange(2), 6)
    groups = [f"g-{index}" for index in range(len(labels))]
    split = {
        group: ("train" if index < 6 else "validation" if index < 10 else "test")
        for index, group in enumerate(groups)
    }
    features = np.zeros((len(labels), 1, 1, 2), dtype=float)
    with pytest.raises(ValueError, match="classifier_backend"):
        evaluate_channel_candidates(
            features,
            labels,
            groups,
            split,
            channel_names=("only",),
            class_names=("healthy", "fault"),
            classifier_backend="sklearn",
        )
