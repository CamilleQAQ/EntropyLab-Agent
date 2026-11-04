"""Classifier algorithm/codec registrations used by training and artifacts.

Registration is deliberately lazy.  A registration describes *how* to build,
fit, save and load one algorithm/backend pair, but no optional backend module is
imported until that registration is actually used.  A machine without PyTorch
can therefore import this registry, train and serve scikit-learn or NumPy
classifiers, and still discover that the Torch backend is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Any, Callable, Mapping

from .lstsvm import LSTSVM
from .obt_lstsvm import OBTLSTSVM
from .sklearn_classifiers import (
    ExtraTreesClassifier,
    RBFSVMClassifier,
    RandomForestClassifier,
)


@dataclass(frozen=True)
class ClassifierRegistration:
    """One algorithm/backend pair and its lazy codec entry points."""

    algorithm: str
    backend: str
    model_type: str
    factory: Callable[[dict[str, Any]], Any]
    fit_model: Callable[[Any, Any, Any, tuple[int, ...]], Any]
    save_model: Callable[[Path, Any, Mapping[str, Any] | None], dict[str, Any]]
    load_model: Callable[[Path], tuple[Any, dict[str, Any]]]
    filename: str = "model.npz"
    requires_trust: bool = False
    requires: tuple[str, ...] = ()
    fixed_kernel: str | None = None

    def configuration(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve configuration through the actual unfit model factory."""
        resolved = dict(self.factory(dict(parameters)).parameters)
        resolved.setdefault("kernel", self.fixed_kernel)
        return resolved

    @property
    def available(self) -> bool:
        """Report optional-dependency availability without importing it."""
        return all(_module_available(name) for name in self.requires)

    @property
    def unavailable_reason(self) -> str | None:
        if self.available:
            return None
        missing = [name for name in self.requires if not _module_available(name)]
        if "torch" in missing:
            extras = "torch"
        elif {"sklearn", "joblib"} & set(missing):
            extras = "ml"
        else:
            extras = None
        install = f"; install entropylab-agent[{extras}]" if extras else ""
        label = "optional dependencies" if extras else "required dependencies"
        return (
            f"{self.algorithm}/{self.backend} requires the {label} "
            f"{', '.join(missing)}{install}"
        )

    def matches(self, model: object) -> bool:
        """Check the concrete class without importing the backend."""
        model_class = type(model)
        return f"{model_class.__module__}:{model_class.__qualname__}" == self.model_type


def _type_name(model_class: type) -> str:
    return f"{model_class.__module__}:{model_class.__qualname__}"


def _module_available(name: str) -> bool:
    """Check an installed module without executing it or importing its parents."""
    try:
        return util.find_spec(name) is not None
    except (ImportError, ValueError):  # broken installation or a None entry
        return False


def _fit_bt(model, x, y, order):
    return model.fit(x, y, class_order=order)


def _fit_obt(model, x, y, order):
    del order  # OBT must derive its order from training rows, never a caller/test set.
    return model.fit(x, y)


def _fit_independent(model, x, y, order):
    del order
    return model.fit(x, y)


def _torch_codec():
    """Import the Torch codec only when the Torch backend is actually used."""
    try:
        from . import torch_artifact
    except ImportError as exc:  # optional dependency, not a programming error
        raise RuntimeError(
            "BT-LSTSVM/torch requires the optional 'torch' dependency; "
            "install entropylab-agent[torch]"
        ) from exc
    return torch_artifact


def _torch_class():
    try:
        from .torch_bt_lstsvm import TorchBTLSTSVM
    except ImportError as exc:  # optional dependency, not a programming error
        raise RuntimeError(
            "BT-LSTSVM/torch requires the optional 'torch' dependency; "
            "install entropylab-agent[torch]"
        ) from exc
    return TorchBTLSTSVM


def _factory_bt(parameters: dict[str, Any]):
    resolved = dict(parameters)
    kernel = resolved.pop("kernel", "rbf")
    if kernel != "rbf":
        raise ValueError("the PyTorch BT-LSTSVM backend currently supports only RBF")
    return _torch_class()(**resolved)


def _save_bt(path, model, metadata):
    return _torch_codec().save_torch_bt_artifact(path, model, metadata=metadata)


def _load_bt(path):
    return _torch_codec().load_torch_bt_artifact(path)


def _save_obt(path, model, metadata):
    from .obt_artifact import save_obt_artifact

    return save_obt_artifact(path, model, metadata=metadata)


def _load_obt(path):
    from .obt_artifact import load_obt_artifact

    return load_obt_artifact(path)


def _save_lstsvm(path, model, metadata):
    from .lstsvm_artifact import save_lstsvm_artifact

    return save_lstsvm_artifact(path, model, metadata=metadata)


def _load_lstsvm(path):
    from .lstsvm_artifact import load_lstsvm_artifact

    return load_lstsvm_artifact(path)


def _save_sklearn(path, model, metadata):
    return model.save(path, metadata=metadata)


def _sklearn_loader(model_class: type):
    def load(path):
        # The pipeline loader must obtain explicit trust before invoking this codec.
        model = model_class.load(path, trusted=True)
        manifest_path = Path(path).with_suffix(Path(path).suffix + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["class_order"] = manifest["classes"]
        return model, manifest

    return load


def _factory_lstsvm(parameters: dict[str, Any]):
    return LSTSVM(**parameters)


def _factory_obt(parameters: dict[str, Any]):
    return OBTLSTSVM(**parameters)


def _sklearn_registration(model_class: type, *, fixed_kernel: str | None = None) -> ClassifierRegistration:
    return ClassifierRegistration(
        algorithm=model_class.algorithm,
        backend="scikit-learn",
        model_type=_type_name(model_class),
        factory=lambda parameters, cls=model_class: cls(**parameters),
        fit_model=_fit_independent,
        save_model=_save_sklearn,
        load_model=_sklearn_loader(model_class),
        filename="model.joblib",
        requires_trust=True,
        requires=("sklearn", "joblib"),
        fixed_kernel=fixed_kernel,
    )


_REGISTRATIONS: dict[tuple[str, str], ClassifierRegistration] = {
    ("LSTSVM", "numpy"): ClassifierRegistration(
        algorithm="LSTSVM",
        backend="numpy",
        model_type=_type_name(LSTSVM),
        factory=_factory_lstsvm,
        fit_model=_fit_independent,
        save_model=_save_lstsvm,
        load_model=_load_lstsvm,
        requires=("numpy", "scipy"),
    ),
    ("BT-LSTSVM", "torch"): ClassifierRegistration(
        algorithm="BT-LSTSVM",
        backend="torch",
        model_type="entropy_agent.classifiers.torch_bt_lstsvm:TorchBTLSTSVM",
        factory=_factory_bt,
        fit_model=_fit_bt,
        save_model=_save_bt,
        load_model=_load_bt,
        requires=("torch",),
        fixed_kernel="rbf",
    ),
    ("OBT-LSTSVM", "numpy"): ClassifierRegistration(
        algorithm="OBT-LSTSVM",
        backend="numpy",
        model_type=_type_name(OBTLSTSVM),
        factory=_factory_obt,
        fit_model=_fit_obt,
        save_model=_save_obt,
        load_model=_load_obt,
        requires=("numpy", "scipy"),
    ),
}
for _model_class, _fixed_kernel in (
    (RBFSVMClassifier, "rbf"),
    (RandomForestClassifier, None),
    (ExtraTreesClassifier, None),
):
    _REGISTRATIONS[(_model_class.algorithm, "scikit-learn")] = (
        _sklearn_registration(_model_class, fixed_kernel=_fixed_kernel)
    )


def classifier_keys() -> tuple[tuple[str, str], ...]:
    return tuple(_REGISTRATIONS)


def classifier_catalog() -> dict[str, dict]:
    """Describe every registered algorithm without importing optional backends."""
    catalog: dict[str, dict] = {}
    for (algorithm, backend), registration in _REGISTRATIONS.items():
        entry: dict[str, Any] = {
            "available": registration.available,
            "backend": backend,
            "requires": list(registration.requires),
        }
        reason = registration.unavailable_reason
        if reason is not None:
            entry["reason"] = reason
        catalog[algorithm] = entry
    return catalog


def get_classifier_registration(
    algorithm: str, backend: str
) -> ClassifierRegistration:
    try:
        return _REGISTRATIONS[(algorithm, backend)]
    except KeyError as exc:
        choices = ", ".join(f"{name}/{kind}" for name, kind in classifier_keys())
        raise ValueError(
            f"unsupported classifier algorithm/backend {algorithm}/{backend}; "
            f"registered choices: {choices}"
        ) from exc


def available_classifier_keys() -> tuple[tuple[str, str], ...]:
    """Only the pairs whose optional dependencies are installed."""
    return tuple(
        key for key, registration in _REGISTRATIONS.items() if registration.available
    )


__all__ = [
    "ClassifierRegistration",
    "available_classifier_keys",
    "classifier_catalog",
    "classifier_keys",
    "get_classifier_registration",
]
