"""Paired, recording-level downsampling and evidence-based conclusion scoring."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from random import Random
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..training.pipeline import evaluate_partition, fit_prepared, prepare_dataset

if TYPE_CHECKING:
    from .tools import LabContext

CONTROL_TOOL = "run_healthy_recall_control"
CONCLUSION_TOOL = "submit_recall_conclusion"
SPLIT_COUNT = 5


class RecallControlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecallConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    baseline_mean_recall: float = Field(ge=0, le=1)
    downsampled_mean_recall: float = Field(ge=0, le=1)
    direction: Literal["decreased", "increased", "inconclusive"]
    scope: Literal["observed_splits"]


def run_healthy_recall_control(ctx: LabContext, _args: RecallControlArgs) -> dict:
    started = perf_counter()
    spec = ctx.spec
    healthy = spec.labels.healthy_class_id
    if healthy is None or spec.split.mode != "random":
        raise ValueError("paired recall control requires a healthy class and a seeded group split")

    pairs = []
    for seed in range(spec.split.seed, spec.split.seed + SPLIT_COUNT):
        seeded = spec.model_copy(update={"split": spec.split.model_copy(update={"seed": seed})})
        prepared = prepare_dataset(seeded, ctx.data_root, include_test=False)
        train = prepared.masks["train"]
        validation = prepared.masks["validation"]
        groups = np.asarray(prepared.groups)
        healthy_records = prepared.labels == healthy
        healthy_train = sorted(set(groups[train & healthy_records]))
        healthy_validation = healthy_records[validation]
        support = int(healthy_validation.sum())
        if len(healthy_train) < 2 or support == 0:
            raise ValueError(f"split {seed} needs two healthy training groups and healthy validation records")
        removed = Random(seed).choice(healthy_train)
        downsampled = replace(prepared, masks={**prepared.masks, "train": train & (groups != removed)})
        pair = {
            "seed": seed,
            "removed_healthy_training_group": removed,
            "healthy_training_groups": healthy_train,
            "validation_groups": sorted(set(groups[validation])),
            "healthy_validation_records": support,
        }
        for name, arm in (("baseline", prepared), ("downsampled", downsampled)):
            model, _ = fit_prepared(arm, seeded)
            _, predicted, _, _ = evaluate_partition(arm, model, "validation", healthy_class_id=healthy)
            correct = int((predicted[healthy_validation] == healthy).sum())
            pair[name] = {"true_positives": correct, "recall": round(correct / support, 6)}
        pairs.append(pair)

    seconds = round(perf_counter() - started, 2)
    ctx.spent_seconds += seconds
    result = {
        "tool": CONTROL_TOOL,
        "evidence_id": uuid4().hex,
        "healthy_label": spec.labels.names[healthy],
        "feature": spec.feature.method,
        "classifier": spec.classifier.algorithm,
        "seeds": len(pairs),
        "per_seed": pairs,
        "seconds": seconds,
    }
    ctx.experiments.append(result)
    return result


def recall_conclusion_checks(rows: list[dict], min_seeds: int) -> dict[str, bool]:
    controls = [row for row in rows if row.get("tool") == CONTROL_TOOL and not row.get("error")]
    submissions = [row for row in rows if row.get("tool") == CONCLUSION_TOOL]
    evidence = None
    if len(controls) == 1:
        evidence = _paired_recalls(controls[0], min_seeds)
    checks = {
        "has_paired_recall_control": evidence is not None,
        "submitted_conclusion": False,
        "conclusion_cites_control": False,
        "conclusion_recall_matches_ledger": False,
        "conclusion_direction_matches_ledger": False,
    }
    if len(submissions) != 1:
        return checks
    try:
        conclusion = RecallConclusion.model_validate(submissions[0].get("conclusion"))
    except ValidationError:
        return checks
    checks["submitted_conclusion"] = True
    if evidence is None or conclusion.evidence_id != controls[0].get("evidence_id"):
        return checks
    checks["conclusion_cites_control"] = True
    baseline, downsampled = evidence
    checks["conclusion_recall_matches_ledger"] = (
        isclose(conclusion.baseline_mean_recall, mean(baseline), abs_tol=0.0001, rel_tol=0)
        and isclose(conclusion.downsampled_mean_recall, mean(downsampled), abs_tol=0.0001, rel_tol=0)
    )
    direction = "inconclusive"
    if all(after < before for before, after in zip(baseline, downsampled)):
        direction = "decreased"
    elif all(after > before for before, after in zip(baseline, downsampled)):
        direction = "increased"
    checks["conclusion_direction_matches_ledger"] = conclusion.direction == direction
    return checks


def _paired_recalls(control, min_seeds):
    baseline, downsampled, seeds = [], [], set()
    try:
        for pair in control["per_seed"]:
            seed = pair["seed"]
            training = set(pair["healthy_training_groups"])
            validation = set(pair["validation_groups"])
            support = pair["healthy_validation_records"]
            before = pair["baseline"]["true_positives"]
            after = pair["downsampled"]["true_positives"]
            if (
                seed in seeds
                or len(training) < 2
                or not validation
                or training & validation
                or pair["removed_healthy_training_group"] not in training
                or not isinstance(support, int)
                or support <= 0
                or any(not isinstance(value, int) or not 0 <= value <= support for value in (before, after))
            ):
                return None
            seeds.add(seed)
            baseline.append(before / support)
            downsampled.append(after / support)
    except (KeyError, TypeError):
        return None
    if len(seeds) < min_seeds or len(seeds) != control.get("seeds"):
        return None
    return baseline, downsampled
