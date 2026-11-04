import numpy as np
import pytest
import torch

from entropy_agent.classifiers.bt_lstsvm import BinaryLSTSVM
from entropy_agent.classifiers.torch_bt_lstsvm import (
    MatlabCompatTorchBTLSTSVM,
    MatlabCompatTorchBinaryLSTSVM,
    MatlabCompatTorchBinaryLSTWSVM,
    TorchBTLSTSVM,
    TorchBTLSTWSVM,
    TorchBinaryLSTSVM,
    TorchBinaryLSTWSVM,
)


def test_torch_backend_is_cpu_float64_and_uses_mkl() -> None:
    assert torch.tensor(0.0, dtype=torch.float64).device.type == "cpu"
    assert torch.backends.mkl.is_available()


def test_historical_torch_names_are_identity_aliases() -> None:
    assert TorchBTLSTWSVM is TorchBTLSTSVM
    assert TorchBinaryLSTWSVM is TorchBinaryLSTSVM
    assert MatlabCompatTorchBinaryLSTWSVM is MatlabCompatTorchBinaryLSTSVM


def test_compatibility_and_deployable_modes_are_explicitly_separated() -> None:
    compat = MatlabCompatTorchBTLSTSVM()
    deployable = TorchBTLSTSVM()
    assert compat.deployment_eligible is False
    assert compat.binary_model_type.include_matlab_zero_rows is True
    assert deployable.deployment_eligible is False
    assert deployable.artifact_registration_capable is True
    assert deployable.binary_model_type.include_matlab_zero_rows is False


def test_deployable_binary_model_does_not_add_zero_rows() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    model = TorchBinaryLSTSVM().fit(x, y)
    assert tuple(model._anchors.shape) == (len(x), x.shape[1])


def test_deployable_tree_prediction_requires_features_only_and_is_deterministic() -> None:
    x = np.array([[-4.0], [-3.0], [0.0], [1.0], [4.0], [5.0]])
    y = np.array([1, 1, 2, 2, 3, 3])
    model = TorchBTLSTSVM().fit(x, y, class_order=(1, 2, 3))
    test = np.array([[-3.5], [0.5], [4.5]])
    first = model.predict(test)
    second = model.predict(test)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3,)
    assert np.isin(first, (1, 2, 3)).all()


def test_tree_trace_records_only_nodes_each_sample_reaches() -> None:
    x = np.array([[-4.0], [-3.0], [0.0], [1.0], [4.0], [5.0]])
    y = np.array([1, 1, 2, 2, 3, 3])
    model = TorchBTLSTSVM().fit(x, y, class_order=(1, 2, 3))
    trace = model.predict_with_trace(np.array([[-3.5], [0.5], [4.5]]))
    np.testing.assert_array_equal(trace.labels, model.predict(np.array([[-3.5], [0.5], [4.5]])))
    assert trace.reached.shape == (3, 2)
    assert trace.accepted_at_node.shape == (3, 2)
    assert np.array_equal(np.isnan(trace.positive_distance), ~trace.reached)
    assert np.array_equal(np.isnan(trace.negative_distance), ~trace.reached)
    assert np.all(trace.reached[:, 0])
    assert np.all(np.sum(trace.accepted_at_node, axis=1) <= 1)


@pytest.mark.parametrize(("positive_rows", "negative_rows"), ((4, 8), (8, 8), (8, 4)))
def test_deployable_torch_nodes_agree_with_numpy_on_well_conditioned_data(
    positive_rows: int, negative_rows: int,
) -> None:
    rng = np.random.default_rng(4200 + positive_rows * 10 + negative_rows)
    positive = rng.normal(-1.0, 0.1, size=(positive_rows, 3))
    negative = rng.normal(1.0, 0.1, size=(negative_rows, 3))
    x = np.vstack((positive, negative))
    y = np.concatenate((np.ones(positive_rows, dtype=int), -np.ones(negative_rows, dtype=int)))
    test = rng.normal(0.0, 1.0, size=(50, 3))
    numpy_model = BinaryLSTSVM(c1=0.25, c2=0.25, bandwidth=4.0).fit(x, y)
    torch_model = TorchBinaryLSTSVM().fit(x, y)
    np.testing.assert_array_equal(torch_model.predict(test), numpy_model.predict(test))
