"""Per-record feature cache keyed by the dataset contract.

The key covers only what determines a feature row: storage layout, input
channel, window, preprocessing and the entropy method with its parameters.
Labels, split and classifier are excluded, so:

* a new split seed reuses every feature row (this is what makes multi-seed
  comparison affordable);
* swapping the classifier is free;
* a leave-one-class-out fold that only renumbers labels still hits the cache.

Rows are stored per record and filled incrementally: asking for a new record
subset extracts only the records that are missing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..io.dataset import DatasetRecord, DatasetSpec
from ..training.pipeline import extract_features

FORMAT_VERSION = 2


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(payload: str) -> str:
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    record_ids: tuple[str, ...]
    dimension: int
    key: str
    hit: bool
    seconds: float
    extracted_records: int

    @property
    def labels(self) -> np.ndarray:
        raise AttributeError("feature rows carry no labels; read them from the records")


class FeatureCache:
    """File-backed per-record cache; ``root=None`` disables persistence."""

    def __init__(self, root: Path | None) -> None:
        self.root = Path(root) if root is not None else None

    # -- keys ------------------------------------------------------------
    def contract_digest(self, spec: DatasetSpec) -> str:
        payload = {
            "dataset_id": spec.dataset_id,
            "input": spec.input.model_dump(mode="json"),
            "layout": spec.layout.model_dump(mode="json"),
            "window": spec.window.model_dump(mode="json"),
            "preprocessing": spec.preprocessing,
            "feature": spec.feature.model_dump(mode="json"),
        }
        return _digest(_canonical(payload))

    def key_for(self, spec: DatasetSpec) -> str:
        slug = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in spec.dataset_id.lower()
        ).strip("-")
        return f"{slug}-{self.contract_digest(spec)}"

    # -- public API ------------------------------------------------------
    def get_or_extract(
        self,
        spec: DatasetSpec,
        data_root: Path,
        records: list[DatasetRecord],
    ) -> FeatureSet:
        key = self.key_for(spec)
        stored = self._load(key, spec) if self.root is not None else None
        mapping: dict[str, np.ndarray] = stored[0] if stored else {}
        wanted = [record.record_id for record in records]
        missing = [record for record in records if record.record_id not in mapping]
        seconds = 0.0
        if missing:
            started = time.perf_counter()
            _, features, _, dimension = extract_features(
                spec, data_root, records=missing
            )
            seconds = time.perf_counter() - started
            for position, record in enumerate(missing):
                mapping[record.record_id] = np.asarray(features[position], dtype=np.float64)
            if self.root is not None:
                self._store(key, spec, mapping)
        rows = [mapping[record_id] for record_id in wanted]
        features = np.stack(rows, axis=0)
        return FeatureSet(
            features=features,
            record_ids=tuple(wanted),
            dimension=int(features.shape[-1]),
            key=key,
            hit=not missing,
            seconds=seconds,
            extracted_records=len(missing),
        )

    # -- storage ---------------------------------------------------------
    def _paths(self, key: str) -> tuple[Path, Path]:
        assert self.root is not None
        return self.root / f"{key}.npz", self.root / f"{key}.json"

    def _load(self, key: str, spec: DatasetSpec):
        array_path, meta_path = self._paths(key)
        if not array_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("format") != FORMAT_VERSION:
                return None
            if meta.get("method") != spec.feature.method:
                return None
            if meta.get("parameters") != dict(spec.feature.parameters):
                return None
            with np.load(array_path) as payload:
                record_ids = [str(value) for value in payload["record_ids"]]
                features = np.asarray(payload["features"], dtype=np.float64)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        if features.ndim != 3 or features.shape[0] != len(record_ids):
            return None
        return (
            {
                record_id: features[index]
                for index, record_id in enumerate(record_ids)
            },
            meta,
        )

    def _store(self, key: str, spec: DatasetSpec, mapping: dict[str, np.ndarray]) -> None:
        assert self.root is not None
        array_path, meta_path = self._paths(key)
        self.root.mkdir(parents=True, exist_ok=True)
        record_ids = list(mapping)
        features = np.stack([mapping[record_id] for record_id in record_ids], axis=0)
        temporary = array_path.with_name(array_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                record_ids=np.asarray(record_ids, dtype=str),
                features=features,
            )
        temporary.replace(array_path)
        meta = {
            "format": FORMAT_VERSION,
            "key": key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": spec.dataset_id,
            "method": spec.feature.method,
            "parameters": dict(spec.feature.parameters),
            "window": spec.window.model_dump(mode="json"),
            "records": len(record_ids),
            "dimension": int(features.shape[-1]),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def count(self) -> int:
        if self.root is None or not self.root.is_dir():
            return 0
        return len(list(self.root.glob("*.npz")))
