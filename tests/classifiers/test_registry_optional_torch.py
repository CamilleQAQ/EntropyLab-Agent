"""The classifier registry must not require PyTorch for non-Torch backends."""

from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest


REGISTRY_MODULE = "entropy_agent.classifiers.registry"


def _purge_optional_backends() -> dict[str, object]:
    saved: dict[str, object] = {}
    for name in list(sys.modules):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name.endswith(".torch_artifact")
            or name.endswith(".torch_bt_lstsvm")
        ):
            saved[name] = sys.modules.pop(name)
    saved[REGISTRY_MODULE] = sys.modules.pop(REGISTRY_MODULE, None)
    return saved


def _restore(saved: dict[str, object]) -> None:
    sys.modules.pop(REGISTRY_MODULE, None)
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("PyTorch is intentionally unavailable for this test")
    return _real_import(name, globals, locals, fromlist, level)


_real_import = builtins.__import__


def test_sklearn_classifier_imports_and_runs_without_torch(monkeypatch) -> None:
    saved = _purge_optional_backends()
    try:
        with monkeypatch.context() as context:
            context.setattr(builtins, "__import__", _blocked_import)
            registry = importlib.import_module(REGISTRY_MODULE)

            # Importing the registry must not pull in the optional backend.
            assert "torch" not in sys.modules
            assert not any(name.endswith(".torch_bt_lstsvm") for name in sys.modules)

            assert registry.classifier_catalog()["SVM-RBF"]["available"] is True
            registration = registry.get_classifier_registration(
                "SVM-RBF", "scikit-learn"
            )
            model = registration.factory({"C": 1.0})
            rng = np.random.default_rng(7)
            features = np.vstack(
                [rng.normal(0.0, 0.1, (12, 3)), rng.normal(1.0, 0.1, (12, 3))]
            )
            labels = np.repeat([0, 1], 12)
            fitted = registration.fit_model(model, features, labels, (0, 1))
            predictions = fitted.predict(features)
            assert set(int(item) for item in predictions) == {0, 1}
            assert "torch" not in sys.modules

            # Asking for the Torch backend fails cleanly and only for that backend.
            with pytest.raises(RuntimeError, match="torch"):
                registry.get_classifier_registration("BT-LSTSVM", "torch").factory({})
            assert registry.get_classifier_registration(
                "LSTSVM", "numpy"
            ).available is True
            assert registry.get_classifier_registration(
                "OBT-LSTSVM", "numpy"
            ).available is True
    finally:
        _restore(saved)


def test_registry_module_does_not_import_torch_in_a_clean_process() -> None:
    """A plain import keeps torch out of ``sys.modules`` until it is used."""
    saved = _purge_optional_backends()
    try:
        registry = importlib.import_module(REGISTRY_MODULE)
        assert "torch" not in sys.modules
        assert registry.classifier_catalog()["RANDOM-FOREST"]["backend"] == "scikit-learn"
    finally:
        _restore(saved)
