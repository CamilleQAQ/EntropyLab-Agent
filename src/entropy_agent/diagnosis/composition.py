"""Application-boundary helpers for assembling optional diagnosis plugins."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from .model_registry import ModelRegistry
from .registry import DiagnosisRegistry


def load_diagnosis_registry(
    provider: str | None,
    options: Mapping[str, Any] | None = None,
    *,
    model_registry_root: Path | None = None,
    trusted: bool = False,
) -> DiagnosisRegistry:
    """Load ``module:factory`` and require it to return a DiagnosisRegistry.

    The generic Agent never imports dataset modules. A deployment chooses a
    dataset/model plugin explicitly at this composition boundary.
    """
    registry = (
        ModelRegistry(model_registry_root, trusted=trusted).as_diagnosis_registry()
        if model_registry_root is not None
        else DiagnosisRegistry()
    )
    if provider is None:
        return registry
    module_name, separator, attribute = provider.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "diagnosis registry provider must use 'module:factory' format"
        )
    module = import_module(module_name)
    factory = getattr(module, attribute)
    provided = factory(**dict(options or {}))
    if not isinstance(provided, DiagnosisRegistry):
        raise TypeError("diagnosis registry provider must return DiagnosisRegistry")
    for profile in provided.profiles():
        registration = provided.get(profile.profile_id)
        if registration is None:
            raise RuntimeError("diagnosis registry changed while composing it")
        registry.register(profile, registration.model_path, registration.runner)
    return registry
