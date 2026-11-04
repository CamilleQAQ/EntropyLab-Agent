"""Configuration-driven labelled signal datasets for reusable training.

The core understands storage layouts, not named datasets. Ordinary datasets
use one signal per file. Packed NPY tensors use explicit axis metadata and an
explicit file-to-label table. Neither path infers a profile from a filename,
sampling rate, or signal statistics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from ..analysis.schemas import StrictModel
from .input_signal import LoadedSignal, SignalInput, load_input_signal


class InputConfig(StrictModel):
    channel: str | int | None = None
    key: str | None = None
    sample_axis: int = -1
    indices: tuple[int, ...] = ()
    has_header: bool = True
    delimiter: str = ","
    skip_rows: int = Field(default=0, ge=0)
    mat_signal_name: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "InputConfig":
        if len(self.delimiter) != 1:
            raise ValueError("input delimiter must contain exactly one character")
        return self


class PackedFile(StrictModel):
    path: str = Field(min_length=1)
    label: str = Field(min_length=1)


class DatasetLayout(StrictModel):
    loader: Literal["files", "npy_tensor"] = "files"
    file_pattern: str = "**/*.csv"
    manifest: str | None = None
    label_source: Literal["parent", "manifest"] = "parent"
    group_source: Literal["file", "parent", "manifest", "regex"] = "file"
    group_regex: str | None = None
    packed_files: tuple[PackedFile, ...] = ()
    record_axis: int = 0
    channel_axis: int = 1
    sample_axis: int = 2
    group_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_layout(self) -> "DatasetLayout":
        if self.loader == "files":
            if self.label_source == "manifest" and not self.manifest:
                raise ValueError("manifest label_source requires layout.manifest")
            if self.group_source == "manifest" and not self.manifest:
                raise ValueError("manifest group_source requires layout.manifest")
            if self.group_source == "regex" and not self.group_regex:
                raise ValueError("regex group_source requires layout.group_regex")
        elif not self.packed_files:
            raise ValueError("npy_tensor loader requires layout.packed_files")
        return self


class LabelConfig(StrictModel):
    names: tuple[str, ...]
    mapping: dict[str, int]
    healthy_class_id: int | None = 0

    @model_validator(mode="after")
    def validate_labels(self) -> "LabelConfig":
        if len(self.names) < 2 or any(not name.strip() for name in self.names):
            raise ValueError("labels.names must contain at least two nonempty classes")
        expected = set(range(len(self.names)))
        if set(self.mapping.values()) != expected:
            raise ValueError("labels.mapping values must cover contiguous class IDs from zero")
        if self.healthy_class_id is not None and self.healthy_class_id not in expected:
            raise ValueError("healthy_class_id is not a configured class")
        return self


class WindowConfig(StrictModel):
    """Window length, step, and the number of windows used per record.

    ``stride`` is an independent step.  It may be smaller than ``size``
    (overlap), equal (non-overlap), or larger (sparse sampling).  Sparse
    sampling is used by the fusion recipes on long Paderborn records.
    """

    size: int = Field(ge=1)
    stride: int = Field(ge=1)
    windows_per_record: int = Field(default=1, ge=1)


class FeatureConfig(StrictModel):
    method: str = Field(min_length=1)
    parameters: dict[str, int | float]


class ClassifierConfig(StrictModel):
    algorithm: str = Field(default="BT-LSTSVM", min_length=1)
    backend: str = Field(default="torch", min_length=1)
    parameters: dict[str, int | float | str | bool | None | dict[str, float]]


class SplitConfig(StrictModel):
    mode: Literal["random", "manifest"] = "random"
    manifest_sha256: str | None = None
    expected_records: dict[str, int] | None = None
    seed: int = 20260928
    train_fraction: float = Field(default=0.60, gt=0, lt=1)
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_split(self) -> "SplitConfig":
        if self.mode == "manifest":
            if not self.manifest_sha256 or not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
                raise ValueError("fixed split requires manifest_sha256")
            if self.expected_records is None or set(self.expected_records) != {
                "train", "validation", "test"
            }:
                raise ValueError(
                    "fixed split requires expected record counts for train, validation and test"
                )
            if any(
                isinstance(value, bool) or value < 0
                for value in self.expected_records.values()
            ):
                raise ValueError("fixed split expected record counts cannot be negative")
            if self.expected_records["train"] < 1 or self.expected_records["validation"] < 1:
                raise ValueError("fixed split requires at least one train and one validation record")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("split fractions must leave a test partition")
        return self


class DiagnosisConfig(StrictModel):
    profile_id: str = Field(min_length=1, pattern=r"^[A-Z0-9_]+$")
    display_name: str = Field(min_length=1)
    workflow: str = "MODEL_ARTIFACT_DIAGNOSIS"
    signal_type: str = Field(min_length=1)
    signal_role: str = Field(min_length=1)
    sampling_rate_tolerance_hz: float = Field(default=0, ge=0)
    minimum_window_count: int = Field(default=1, ge=1)
    evidence_window_count: int = Field(default=1, ge=1)
    validation_scope: str = Field(min_length=1)
    experimental: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> "DiagnosisConfig":
        if self.minimum_window_count > self.evidence_window_count:
            raise ValueError("minimum_window_count cannot exceed evidence_window_count")
        return self


class OODConfig(StrictModel):
    calibration: Literal["DISABLED", "VALIDATION_MAX_NEAREST_TRAINING"] = "DISABLED"
    margin: float = Field(default=1.05, ge=1, allow_inf_nan=False)


class DatasetSpec(StrictModel):
    """All dataset-dependent facts required by the generic training pipeline."""

    format: Literal["entropylab.dataset-spec"] = "entropylab.dataset-spec"
    format_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False)
    input: InputConfig
    layout: DatasetLayout
    labels: LabelConfig
    window: WindowConfig
    preprocessing: Literal["NONE"] = "NONE"
    feature: FeatureConfig
    classifier: ClassifierConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    diagnosis: DiagnosisConfig
    ood: OODConfig = Field(default_factory=OODConfig)

    @model_validator(mode="after")
    def validate_stride_contract(self) -> "DatasetSpec":
        """Keep the single-entropy contract unambiguous.

        Non-fusion methods use ``window.stride`` as the only window step.  A
        second ``sample_stride`` parameter would silently override it and make
        the artifact metadata disagree with the actual features.  Fusion
        recipes may carry their explicit sampling stride later; that contract
        will get its own effective-stride field instead of reusing this one.
        """
        if not self.feature.method.startswith("FUSION_") and "sample_stride" in self.feature.parameters:
            raise ValueError(
                "feature.parameters.sample_stride is only valid for FUSION_* methods; "
                "use window.stride for single-entropy methods"
            )
        return self


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    label: int
    group: str
    signal_input: SignalInput
    partition: str | None = None


def load_dataset_spec(path: Path) -> DatasetSpec:
    """Load a strict JSON dataset specification without executing code."""
    try:
        value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load dataset specification: {exc}") from exc
    return DatasetSpec.model_validate(value, strict=False)


def _signal_input(path: Path, config: InputConfig, sampling_rate_hz: float | None = None) -> SignalInput:
    return SignalInput(
        path=path,
        channel=config.channel,
        key=config.key,
        sample_axis=config.sample_axis,
        indices=config.indices,
        has_header=config.has_header,
        delimiter=config.delimiter,
        skip_rows=config.skip_rows,
        mat_signal_name=config.mat_signal_name,
        nominal_sampling_rate_hz=sampling_rate_hz if config.mat_signal_name else None,
    )


def _label_id(token: str, spec: DatasetSpec) -> int:
    try:
        return spec.labels.mapping[token]
    except KeyError as exc:
        raise ValueError(
            f"label token {token!r} is not declared in labels.mapping"
        ) from exc


def _manifest_records(spec: DatasetSpec, root: Path) -> list[DatasetRecord]:
    manifest_path = (root / str(spec.layout.manifest)).resolve()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ValueError(f"cannot read dataset manifest: {exc}") from exc
    required = {"path", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("dataset manifest requires path and label columns")
    if spec.split.mode == "manifest":
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != spec.split.manifest_sha256:
            raise ValueError("fixed split manifest SHA-256 changed")
        if not {"record_id", "partition"}.issubset(rows[0]):
            raise ValueError("fixed split manifest requires record_id and partition")
    records = []
    for index, row in enumerate(rows):
        relative = row["path"].strip()
        if not relative:
            raise ValueError(f"manifest row {index + 2} has an empty path")
        path = (root / relative).resolve()
        group = row.get("group", "").strip() or relative
        records.append(
            DatasetRecord(
                record_id=row.get("record_id", "").strip() or relative,
                label=_label_id(row["label"].strip(), spec),
                group=group,
                signal_input=_signal_input(path, spec.input, spec.sampling_rate_hz),
                partition=row.get("partition") if spec.split.mode == "manifest" else None,
            )
        )
    return records


def _file_records(spec: DatasetSpec, root: Path) -> list[DatasetRecord]:
    if spec.layout.manifest:
        return _manifest_records(spec, root)
    records = []
    pattern = re.compile(spec.layout.group_regex) if spec.layout.group_regex else None
    for path in sorted(root.glob(spec.layout.file_pattern)):
        if not path.is_file() or path.parent.name not in spec.labels.mapping:
            continue
        relative = path.relative_to(root).as_posix()
        if spec.layout.group_source == "parent":
            group = path.parent.as_posix()
        elif spec.layout.group_source == "regex":
            match = pattern.search(relative) if pattern is not None else None
            if match is None:
                raise ValueError(f"group_regex does not match {relative}")
            group = match.groupdict().get("group") or match.group(0)
        else:
            group = relative
        records.append(
            DatasetRecord(
                record_id=relative,
                label=_label_id(path.parent.name, spec),
                group=group,
                signal_input=_signal_input(path.resolve(), spec.input, spec.sampling_rate_hz),
            )
        )
    return records


def _packed_records(spec: DatasetSpec, root: Path) -> list[DatasetRecord]:
    layout = spec.layout
    axes = (layout.record_axis, layout.channel_axis, layout.sample_axis)
    if len(set(axes)) != 3 or set(axes) != {0, 1, 2}:
        raise ValueError("npy_tensor record/channel/sample axes must be a permutation of 0,1,2")
    if not isinstance(spec.input.channel, int):
        raise ValueError("npy_tensor input.channel must be an integer channel index")
    records = []
    for packed in layout.packed_files:
        path = (root / packed.path).resolve()
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot inspect packed NPY {path}: {exc}") from exc
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"packed NPY must be a numeric 3-D tensor: {path}")
        channel = spec.input.channel
        if channel < 0 or channel >= array.shape[layout.channel_axis]:
            raise ValueError(f"channel {channel} is outside packed tensor {path}")
        record_count = array.shape[layout.record_axis]
        other_axes = [axis for axis in range(3) if axis != layout.sample_axis]
        for record_index in range(record_count):
            by_axis = {
                layout.record_axis: record_index,
                layout.channel_axis: channel,
            }
            indices = tuple(by_axis[axis] for axis in other_axes)
            record_id = f"{packed.path}#{record_index}"
            records.append(
                DatasetRecord(
                    record_id=record_id,
                    label=_label_id(packed.label, spec),
                    group=(
                        f"{packed.path}#group-"
                        f"{record_index // layout.group_size}"
                    ),
                    signal_input=SignalInput(
                        path=path,
                        sample_axis=layout.sample_axis,
                        indices=indices,
                    ),
                )
            )
    return records


def load_dataset_records(spec: DatasetSpec, root: Path) -> list[DatasetRecord]:
    """Inventory labelled records using only the explicit dataset contract."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    records = (
        _file_records(spec, root)
        if spec.layout.loader == "files"
        else _packed_records(spec, root)
    )
    if not records:
        raise ValueError("dataset specification selected no records")
    present = {record.label for record in records}
    expected = set(range(len(spec.labels.names)))
    if present != expected:
        raise ValueError(f"dataset labels {sorted(present)} do not cover {sorted(expected)}")
    if spec.split.mode == "manifest":
        resolve_dataset_split(spec, records)
    return records


def load_record_signal(record: DatasetRecord) -> LoadedSignal:
    return load_input_signal(record.signal_input)


def resolve_dataset_split(spec: DatasetSpec, records: list[DatasetRecord]) -> dict[str, str]:
    """Resolve record metadata before any signal/window is read, with no fixed-split fallback."""
    if spec.split.mode == "random":
        from ..training.grouped import stratified_group_split
        return stratified_group_split(
            [r.label for r in records], [r.group for r in records],
            seed=spec.split.seed, train_fraction=spec.split.train_fraction,
            validation_fraction=spec.split.validation_fraction,
        )
    from collections import Counter
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("duplicate original record identity in fixed manifest")
    by_group: dict[str, str] = {}
    by_path: dict[tuple, str] = {}
    counts = Counter()
    for r in records:
        if r.partition not in {"train", "validation", "test"}:
            raise ValueError(f"invalid fixed partition for {r.record_id}: {r.partition}")
        identity = (r.signal_input.path.resolve(), r.signal_input.indices)
        if identity in by_path:
            raise ValueError("original signal appears more than once in fixed manifest")
        by_path[identity] = r.partition
        if r.group in by_group and by_group[r.group] != r.partition:
            raise ValueError("original group crosses fixed partitions")
        by_group[r.group] = r.partition
        counts[r.partition] += 1
    actual_counts = {
        name: int(counts.get(name, 0))
        for name in ("train", "validation", "test")
    }
    if actual_counts != spec.split.expected_records:
        raise ValueError(
            f"fixed split record counts {actual_counts} differ from "
            f"{spec.split.expected_records}"
        )
    return by_group
