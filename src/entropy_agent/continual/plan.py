"""Batch plan for continual learning.

A plan is an explicit, auditable partition of the dataset's groups into:

* ``batches`` - ordered incoming data blocks (batch 0 is the initial training set);
* ``validation_groups`` - one frozen selection set, used after every batch;
* ``holdout_groups`` - one frozen final set, scored exactly once at the end.

Groups are the dataset's own grouping unit (one file, or one consecutive block of
windows), so a plan never depends on a dataset name.  Validation and holdout are
part of the plan, not of the incoming batches: a group belongs to exactly one
place, which is what makes "history never gets re-split" enforceable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "entropylab.continual-plan"
FORMAT_VERSION = 1


class PlanError(ValueError):
    """The plan does not partition the dataset in an auditable way."""


@dataclass(frozen=True)
class Batch:
    batch_id: int
    groups: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class BatchPlan:
    dataset_id: str
    batches: tuple[Batch, ...]
    validation_groups: tuple[str, ...]
    holdout_groups: tuple[str, ...]
    margin: float = 0.005
    notes: str = ""

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict) -> "BatchPlan":
        if payload.get("format") != FORMAT:
            raise PlanError(
                f"plan format must be {FORMAT!r}, got {payload.get('format')!r}"
            )
        if payload.get("format_version") != FORMAT_VERSION:
            raise PlanError(
                f"plan format_version must be {FORMAT_VERSION}, got {payload.get('format_version')!r}"
            )
        batches = tuple(
            Batch(
                batch_id=int(entry["batch_id"]),
                groups=tuple(str(group) for group in entry["groups"]),
                note=str(entry.get("note", "")),
            )
            for entry in payload["batches"]
        )
        ids = [batch.batch_id for batch in batches]
        if ids != list(range(len(batches))):
            raise PlanError(f"batch ids must be 0..N-1 in order, got {ids}")
        margin = float(payload.get("margin", 0.005))
        if not 0.0 <= margin < 1.0:
            raise PlanError("margin must be in [0, 1)")
        return cls(
            dataset_id=str(payload["dataset_id"]),
            batches=batches,
            validation_groups=tuple(str(g) for g in payload["validation_groups"]),
            holdout_groups=tuple(str(g) for g in payload["holdout_groups"]),
            margin=margin,
            notes=str(payload.get("notes", "")),
        )

    @classmethod
    def load(cls, path: Path) -> "BatchPlan":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PlanError(f"cannot read plan: {error}") from error
        return cls.from_dict(payload)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "dataset_id": self.dataset_id,
            "margin": self.margin,
            "notes": self.notes,
            "batches": [
                {"batch_id": batch.batch_id, "groups": list(batch.groups), "note": batch.note}
                for batch in self.batches
            ],
            "validation_groups": list(self.validation_groups),
            "holdout_groups": list(self.holdout_groups),
        }

    # -- properties ------------------------------------------------------
    @property
    def batch_count(self) -> int:
        return len(self.batches)

    def groups_of(self, batch_index: int) -> tuple[str, ...]:
        return self.batches[batch_index].groups

    def history_groups(self, batch_index: int) -> tuple[str, ...]:
        """Every group seen from batch 0 through ``batch_index`` (never re-split)."""
        if not 0 <= batch_index < self.batch_count:
            raise PlanError(f"batch index {batch_index} out of range")
        collected: list[str] = []
        for batch in self.batches[: batch_index + 1]:
            collected.extend(batch.groups)
        return tuple(collected)

    # -- validation ------------------------------------------------------
    def validate(self, records) -> dict:
        """Check the plan partitions the dataset and covers every class."""
        record_groups = [record.group for record in records]
        label_by_group = {}
        for record in records:
            label_by_group.setdefault(record.group, int(record.label))

        planned = (
            self.validation_groups
            + self.holdout_groups
            + tuple(group for batch in self.batches for group in batch.groups)
        )
        if len(set(planned)) != len(planned):
            duplicates = sorted(
                group for group in set(planned) if planned.count(group) > 1
            )
            raise PlanError(f"groups appear more than once in the plan: {duplicates[:5]}")

        known = set(record_groups)
        unknown = sorted(set(planned) - known)
        if unknown:
            raise PlanError(f"plan references unknown groups: {unknown[:5]}")
        missing = sorted(known - set(planned))
        if missing:
            raise PlanError(f"plan leaves groups unassigned: {missing[:5]}")

        classes = sorted(set(label_by_group.values()))
        for name, groups in (
            ("validation", self.validation_groups),
            ("holdout", self.holdout_groups),
        ):
            covered = {label_by_group[group] for group in groups}
            if covered != set(classes):
                absent = sorted(set(classes) - covered)
                raise PlanError(f"{name} does not cover classes {absent}")
        initial = {label_by_group[group] for group in self.batches[0].groups}
        return {
            "batches": self.batch_count,
            "groups": {
                "batches": [len(batch.groups) for batch in self.batches],
                "validation": len(self.validation_groups),
                "holdout": len(self.holdout_groups),
            },
            "classes": classes,
            "initial_batch_covers_all_classes": initial == set(classes),
            "initial_batch_missing_classes": sorted(set(classes) - initial),
        }

    # -- masks -----------------------------------------------------------
    def masks(self, records, batch_index: int) -> dict[str, np.ndarray]:
        """Train on history through ``batch_index``; validation frozen; test = holdout."""
        groups = np.asarray([record.group for record in records], dtype=object)
        history = np.isin(groups, np.asarray(self.history_groups(batch_index), dtype=object))
        validation = np.isin(groups, np.asarray(self.validation_groups, dtype=object))
        holdout = np.isin(groups, np.asarray(self.holdout_groups, dtype=object))
        if history.any() and (history & validation).any():
            raise PlanError("validation groups leaked into the training history")
        if history.any() and (history & holdout).any():
            raise PlanError("holdout groups leaked into the training history")
        return {
            "train": history,
            "validation": validation,
            "test": holdout,
        }


def save_plan(path: Path, plan: BatchPlan) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
