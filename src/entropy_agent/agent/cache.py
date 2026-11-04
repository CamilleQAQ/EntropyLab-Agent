"""Feature cache keyed by the dataset contract, never by a dataset name.

The key covers the parts of a ``DatasetSpec`` that determine the feature matrix
(storage layout, input channel, labels, window, preprocessing, feature method and
parameters) plus the identity of the record subset.  It deliberately excludes the
split and the classifier, so changing a classifier or a seed reuses the features
and only a new entropy method/parameter set costs an extraction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..io.dataset import DatasetRecord, DatasetSpec
from ..training.pipeline import extract_features


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _short_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray
    record_ids: tuple[str, ...]
    groups: tuple[str, ...]
    dimension: int
    key: str
    hit: bool
    seconds: float


class FeatureCache:
    """File-backed cache; ``root=None`` disables persistence."""

    def __init__(self, root: Path | None) -> None:
        self.root = Path(root) if root is not None else None

    def contract_digest(self, spec: DatasetSpec) -> str:
        payload = {
            "dataset_id": spec.dataset_id,
            "input": spec.input.model_dump(mode="json"),
            "layout": spec.layout.model_dump(mode="json"),
            "labels": spec.labels.model_dump(mode="json"),
            "window": spec.window.model_dump(mode="json"),
            "preprocessing": spec.preprocessing,
            "feature": spec.feature.model_dump(mode="json"),
        }
        return _short_digest(_canonical(payload))

    def records_digest(self, records: list[DatasetRecord]) -> str:
        payload = [(record.record_id, int(record.label), record.group) for record in records]
        return _short_digest(_canonical(payload))

    def key_for(self, spec: DatasetSpec, records: list[DatasetRecord]) -> str:
        slug = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in spec.dataset_id.lower()
        ).strip("-")
        return f"{slug}-{self.contract_digest(spec)}-{self.records_digest(records)}"

    def get_or_extract(
        self,
        spec: DatasetSpec,
        data_root: Path,
        records: list[DatasetRecord],
    ) -> FeatureSet:
        key = self.key_for(spec, records)
        if self.root is not None:
            cached = self._load(key, spec, records)
            if cached is not None:
                return cached
        started = _clock()
        _, features, labels, dimension = extract_features(spec, data_root, records=records)
        seconds = _clock() - started
        result = FeatureSet(
            features=features,
            labels=labels,
            record_ids=tuple(record.record_id for record in records),
            groups=tuple(record.group for record in records),
            dimension=int(dimension),
            key=key,
            hit=False,
            seconds=seconds,
        )
        if self.root is not None:
            self._store(result, spec)
        return result

    # -- storage ---------------------------------------------------------
    def _paths(self, key: str) -> tuple[Path, Path]:
        assert self.root is not None
        return self.root / f"{key}.npz", self.root / f"{key}.json"

    def _load(
        self, key: str, spec: DatasetSpec, records: list[DatasetRecord]
    ) -> FeatureSet | None:
        array_path, meta_path = self._paths(key)
        if not array_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(array_path) as payload:
                features = np.asarray(payload["features"], dtype=np.float64)
                labels = np.asarray(payload["labels"], dtype=np.int64)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        expected_labels = np.asarray([int(record.label) for record in records], dtype=np.int64)
        if labels.shape != expected_labels.shape or not np.array_equal(labels, expected_labels):
            return None
        if features.ndim != 3 or features.shape[0] != len(records):
            return None
        if meta.get("method") != spec.feature.method:
            return None
        if meta.get("parameters") != dict(spec.feature.parameters):
            return None
        return FeatureSet(
            features=features,
            labels=labels,
            record_ids=tuple(record.record_id for record in records),
            groups=tuple(record.group for record in records),
            dimension=int(features.shape[-1]),
            key=key,
            hit=True,
            seconds=0.0,
        )

    def _store(self, result: FeatureSet, spec: DatasetSpec) -> None:
        assert self.root is not None
        array_path, meta_path = self._paths(result.key)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_name(array_path.name + ".tmp")
        with temporary.open("wb") as handle:
            # A file object avoids numpy appending another ".npz" suffix.
            np.savez_compressed(handle, features=result.features, labels=result.labels)
        temporary.replace(array_path)
        meta = {
            "key": result.key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": spec.dataset_id,
            "method": spec.feature.method,
            "parameters": dict(spec.feature.parameters),
            "window": spec.window.model_dump(mode="json"),
            "records": len(result.record_ids),
            "dimension": result.dimension,
            "extract_seconds": result.seconds,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _clock() -> float:
    import time

    return time.perf_counter()
