"""What the Agent is allowed to know about a dataset before any trial runs.

The profile is derived from the injected ``DatasetSpec`` and the resolved split
only.  It never infers a dataset, a machine or a sampling rate from signal
statistics, and it carries no dataset-specific prose.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from ..io.dataset import DatasetRecord, DatasetSpec


@dataclass(frozen=True)
class DataProfile:
    dataset_id: str
    class_count: int
    class_names: tuple[str, ...]
    healthy_class_id: int | None
    window_size: int
    windows_per_record: int
    split_mode: str
    records_per_split: dict[str, int]
    records_per_class: dict[str, int]
    group_count: int
    sampling_rate_hz: float

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["class_names"] = list(self.class_names)
        return payload


def build_profile(
    spec: DatasetSpec,
    records: list[DatasetRecord],
    split_by_group: dict[str, str],
) -> DataProfile:
    split_counts = Counter(split_by_group[record.group] for record in records)
    class_counts = Counter(spec.labels.names[record.label] for record in records)
    return DataProfile(
        dataset_id=spec.dataset_id,
        class_count=len(spec.labels.names),
        class_names=tuple(spec.labels.names),
        healthy_class_id=spec.labels.healthy_class_id,
        window_size=spec.window.size,
        windows_per_record=spec.window.windows_per_record,
        split_mode=spec.split.mode,
        records_per_split={
            name: int(split_counts.get(name, 0))
            for name in ("train", "validation", "test")
        },
        records_per_class=dict(sorted(class_counts.items())),
        group_count=len(split_by_group),
        sampling_rate_hz=float(spec.sampling_rate_hz),
    )
