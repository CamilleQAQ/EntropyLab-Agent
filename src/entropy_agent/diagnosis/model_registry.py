"""Persistent index of self-contained model directories.

The registry is deliberately not the model. Removing an index entry leaves the
artifact intact unless file deletion is explicitly requested.
"""

from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

from .artifact import ModelArtifact, load_model_artifact
from .registry import DiagnosisRegistry
from .runner import diagnose_artifact_signal


REGISTRY_FORMAT = "entropylab.model-registry"
REGISTRY_VERSION = 1


class ModelRegistry:
    def __init__(self, root: Path, *, trusted: bool = False) -> None:
        self.trusted = trusted
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "registry.json"

    def _empty(self) -> dict[str, Any]:
        return {
            "format": REGISTRY_FORMAT,
            "format_version": REGISTRY_VERSION,
            "models": {},
        }

    def _read(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty()
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read model registry: {exc}") from exc
        if (
            not isinstance(value, dict)
            or value.get("format") != REGISTRY_FORMAT
            or value.get("format_version") != REGISTRY_VERSION
            or not isinstance(value.get("models"), dict)
        ):
            raise ValueError("model registry format is incompatible")

        return value

    def _write(self, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name(self.index_path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.index_path)

    def _stored_path(self, directory: Path) -> str:
        try:
            return directory.relative_to(self.root).as_posix()
        except ValueError:
            return str(directory)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return (self.root / path).resolve() if not path.is_absolute() else path.resolve()

    def register(self, artifact_directory: Path, *, replace: bool = False) -> dict[str, Any]:
        artifact = load_model_artifact(
            artifact_directory,
            trusted=self.trusted,
        )
        model_id = artifact.config["model_id"]
        index = self._read()
        if model_id in index["models"] and not replace:
            raise ValueError(f"model {model_id!r} is already registered")

        index["models"][model_id] = {"path": self._stored_path(artifact.directory)}
        self._write(index)
        return self.show(model_id)

    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._read()["models"]))

    def artifact_path(self, model_id: str) -> Path:
        registry_entry = self._read()["models"].get(model_id)
        if not isinstance(registry_entry, dict) or not isinstance(
            registry_entry.get("path"), str
        ):
            raise ValueError(f"model {model_id!r} is not registered")

        return self._resolve_path(registry_entry["path"])

    def load(self, model_id: str) -> ModelArtifact:
        return load_model_artifact(
            self.artifact_path(model_id),
            trusted=self.trusted,
        )

    def show(self, model_id: str) -> dict[str, Any]:
        artifact = self.load(model_id)
        return {
            "model_id": model_id,
            "path": str(artifact.directory),
            "format_version": artifact.config["format_version"],
            "profile": artifact.profile.model_dump(mode="json"),
            "feature": artifact.config["feature"],
            "classifier": artifact.config["classifier"],
            "provenance": artifact.config["provenance"],
            "test_result": artifact.test_result,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.show(model_id) for model_id in self.model_ids()]

    def delete(self, model_id: str, *, delete_files: bool = False) -> Path:
        index = self._read()
        if model_id not in index["models"]:
            raise ValueError(f"model {model_id!r} is not registered")
        artifact_path = self._resolve_path(index["models"][model_id]["path"])
        if delete_files:
            try:
                artifact_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(
                    "refusing to delete an artifact outside the registry root"
                ) from exc
            if artifact_path == self.root:
                raise ValueError("refusing to delete the registry root")

        del index["models"][model_id]
        self._write(index)
        if delete_files:
            if artifact_path.exists():
                shutil.rmtree(artifact_path)

        return artifact_path

    def as_diagnosis_registry(self) -> DiagnosisRegistry:
        registry = DiagnosisRegistry()
        for model_id in self.model_ids():
            artifact = self.load(model_id)
            registry.register(
                artifact.profile,
                artifact.directory,
                partial(
                    diagnose_artifact_signal,
                    trusted=self.trusted,
                ),
            )
        return registry
