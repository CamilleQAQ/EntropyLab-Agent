"""Continual-learning loop: one frozen configuration, incoming batches, champion/challenger.

The configuration (feature method + classifier) is fixed for the whole run: the
Agent selected it once, and continual learning only adds data.  That keeps the
experiment about *data accumulation*, not about re-selection noise on tiny
per-batch validation sets.

Rules, in one place:

* training set at batch ``t`` = every batch from 0 through ``t``; it only grows;
* the validation set is frozen and used for every promotion decision;
* the holdout set is scored exactly once, after the last batch, for the three arms
  (frozen / continual / joint);
* a challenger replaces the champion only when it wins by at least ``plan.margin``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.dataset import DatasetSpec, load_dataset_records
from ..training.pipeline import (
    PreparedDataset,
    evaluate_partition,
    extract_features,
    train_dataset,
)
from .plan import BatchPlan


@dataclass
class Champion:
    batch_id: int
    model_id: str
    model_directory: Path
    model: object
    validation_macro_f1: float
    validation_accuracy: float


def _prepared(
    spec: DatasetSpec,
    records,
    features: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    masks: dict[str, np.ndarray],
) -> PreparedDataset:
    split_by_group: dict[str, str] = {}
    for record, name in zip(records, _names(masks)):
        split_by_group.setdefault(record.group, name)
    return PreparedDataset(
        spec=spec,
        records=list(records),
        features=features,
        labels=labels,
        groups=tuple(record.group for record in records),
        feature_dimension=dimension,
        split_by_group=split_by_group,
        masks=masks,
        feature_seconds=0.0,
    )


def _names(masks: dict[str, np.ndarray]):
    names = ["train"] * len(next(iter(masks.values())))
    for split_name in ("validation", "test"):
        mask = masks[split_name]
        for index in np.flatnonzero(mask):
            names[int(index)] = split_name
    return names


def _decision(
    batch_id: int,
    challenger: dict,
    champion: Champion | None,
    promoted: bool,
    reason: str,
) -> dict:
    return {
        "batch_id": batch_id,
        "challenger": {
            "model_id": challenger["model_id"],
            "validation_macro_f1": round(float(challenger["val_macro_f1"]), 6),
            "validation_accuracy": round(float(challenger["val_accuracy"]), 6),
            "training_records": int(challenger["training_records"]),
        },
        "champion_before": (
            None
            if champion is None
            else {
                "batch_id": champion.batch_id,
                "model_id": champion.model_id,
                "validation_macro_f1": round(float(champion.validation_macro_f1), 6),
            }
        ),
        "promoted": promoted,
        "reason": reason,
    }


def run_continual(
    spec: DatasetSpec,
    data_root: Path,
    plan: BatchPlan,
    output_dir: Path,
    *,
    model_prefix: str = "continual",
    run_holdout: bool = True,
) -> dict:
    if plan.dataset_id != spec.dataset_id:
        raise ValueError(
            f"plan is for {plan.dataset_id!r} but the spec is {spec.dataset_id!r}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_dataset_records(spec, data_root)
    plan_info = plan.validate(records)
    records, features, labels, dimension = extract_features(
        spec, data_root, records=records
    )
    labels = np.asarray(labels, dtype=np.int64)
    healthy = spec.labels.healthy_class_id

    decisions: list[dict] = []
    champion: Champion | None = None
    frozen: Champion | None = None
    trajectory: list[dict] = []

    for batch_index in range(plan.batch_count):
        masks = plan.masks(records, batch_index)
        prepared = _prepared(spec, records, features, labels, dimension, masks)
        model_id = f"{model_prefix}-{spec.dataset_id.lower()}-batch-{batch_index}"
        result = train_dataset(
            spec,
            Path(data_root),
            output_dir / "challengers" / f"batch-{batch_index}",
            model_id=model_id,
            prepared=prepared,
            evaluate_test=False,
        )
        validation = result.test_result["validation"]
        challenger = {
            "model_id": model_id,
            "val_macro_f1": float(validation["multiclass_macro_f1"]),
            "val_accuracy": float(validation["multiclass_accuracy"]),
            "training_records": int(masks["train"].sum()),
        }

        if champion is None:
            promoted, reason = True, "initial batch: the first model becomes the champion"
        else:
            threshold = champion.validation_macro_f1 + plan.margin
            promoted = challenger["val_macro_f1"] >= threshold
            reason = (
                f"{challenger['val_macro_f1']:.4f} vs {champion.validation_macro_f1:.4f} "
                f"(needs {threshold:.4f} = champion + margin {plan.margin})"
            )
        decisions.append(_decision(batch_index, challenger, champion, promoted, reason))

        if promoted:
            champion = Champion(
                batch_id=batch_index,
                model_id=model_id,
                model_directory=output_dir / "challengers" / f"batch-{batch_index}",
                model=result.artifact.model,
                validation_macro_f1=challenger["val_macro_f1"],
                validation_accuracy=challenger["val_accuracy"],
            )
            if frozen is None:
                frozen = champion
        trajectory.append(
            {
                "batch_id": batch_index,
                "note": plan.batches[batch_index].note,
                "training_records": challenger["training_records"],
                "challenger_macro_f1": round(challenger["val_macro_f1"], 6),
                "champion_macro_f1": round(champion.validation_macro_f1, 6),
                "promoted": promoted,
            }
        )

    # Joint arm: the same configuration retrained on every batch at once.
    all_groups = tuple(group for batch in plan.batches for group in batch.groups)
    joint_masks = {
        "train": np.isin(np.asarray([r.group for r in records], dtype=object), all_groups),
        "validation": np.isin(
            np.asarray([r.group for r in records], dtype=object), plan.validation_groups
        ),
        "test": np.isin(
            np.asarray([r.group for r in records], dtype=object), plan.holdout_groups
        ),
    }
    joint_prepared = _prepared(spec, records, features, labels, dimension, joint_masks)
    joint_result = train_dataset(
        spec,
        Path(data_root),
        output_dir / "joint",
        model_id=f"{model_prefix}-{spec.dataset_id.lower()}-joint",
        prepared=joint_prepared,
        evaluate_test=False,
    )
    joint = Champion(
        batch_id=plan.batch_count - 1,
        model_id=joint_result.artifact.config["model_id"],
        model_directory=output_dir / "joint",
        model=joint_result.artifact.model,
        validation_macro_f1=float(joint_result.test_result["validation"]["multiclass_macro_f1"]),
        validation_accuracy=float(joint_result.test_result["validation"]["multiclass_accuracy"]),
    )

    holdout = None
    if run_holdout:
        # One test phase, three already-frozen models.
        holdout = {}
        for name, model in (
            ("frozen", frozen.model if frozen else None),
            ("continual", champion.model if champion else None),
            ("joint", joint.model),
        ):
            if model is None:
                continue
            metrics, _, _, _ = evaluate_partition(
                joint_prepared, model, "test", healthy_class_id=healthy
            )
            holdout[name] = {
                "accuracy": float(metrics["multiclass_accuracy"]),
                "macro_f1": float(metrics["multiclass_macro_f1"]),
                "records": int(joint_masks["test"].sum()),
            }

    summary = {
        "dataset_id": spec.dataset_id,
        "configuration": {
            "feature": spec.feature.method,
            "feature_dimension": int(dimension),
            "classifier": spec.classifier.algorithm,
        },
        "plan": plan_info,
        "margin": plan.margin,
        "batches": [
            {"batch_id": batch.batch_id, "note": batch.note, "groups": len(batch.groups)}
            for batch in plan.batches
        ],
        "decisions": decisions,
        "trajectory": trajectory,
        "promotions": sum(1 for decision in decisions if decision["promoted"]),
        "final_champion": None
        if champion is None
        else {
            "batch_id": champion.batch_id,
            "model_id": champion.model_id,
            "validation_macro_f1": round(champion.validation_macro_f1, 6),
            "model_directory": str(champion.model_directory),
        },
        "joint_validation_macro_f1": round(joint.validation_macro_f1, 6),
        "holdout": holdout,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary, trajectory, holdout)
    return summary


def _write_report(path: Path, summary: dict, trajectory: list[dict], holdout: dict | None) -> None:
    lines = [
        "# 持续学习运行报告",
        "",
        f"- 数据集：{summary['dataset_id']}",
        f"- 固定配置：{summary['configuration']['feature']} + {summary['configuration']['classifier']}"
        f"（{summary['configuration']['feature_dimension']} 维）",
        f"- 批次：{summary['plan']['batches']} 个；promote margin {summary['margin']}",
        f"- 组数：批次 {summary['plan']['groups']['batches']}，"
        f"validation {summary['plan']['groups']['validation']}，"
        f"holdout {summary['plan']['groups']['holdout']}",
        "",
        "## 1. 逐批决策",
        "",
        "| 批次 | 说明 | 训练记录 | challenger macro-F1 | champion macro-F1 | 决定 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in trajectory:
        decision = "promote" if row["promoted"] else "reject"
        lines.append(
            f"| {row['batch_id']} | {row['note']} | {row['training_records']} | "
            f"{row['challenger_macro_f1']:.4f} | {row['champion_macro_f1']:.4f} | {decision} |"
        )
    lines.append("")
    lines.append("## 2. 判定理由")
    lines.append("")
    for decision in summary["decisions"]:
        lines.append(
            f"- 批次 {decision['batch_id']}：{decision['reason']} → "
            f"{'promote' if decision['promoted'] else 'reject'}"
        )
    lines.append("")
    lines.append("## 3. 固定 holdout 三臂对比（只评一次）")
    lines.append("")
    if holdout:
        lines.append("| 臂 | accuracy | macro-F1 | 记录数 |")
        lines.append("|---|---:|---:|---:|")
        for name in ("frozen", "continual", "joint"):
            entry = holdout.get(name)
            if entry is None:
                continue
            lines.append(
                f"| {name} | {entry['accuracy']:.4f} | {entry['macro_f1']:.4f} | {entry['records']} |"
            )
        frozen = holdout.get("frozen")
        continual = holdout.get("continual")
        joint = holdout.get("joint")
        lines.append("")
        if frozen and continual:
            lines.append(
                f"- 持续学习相对 frozen 的变化："
                f"{continual['accuracy'] - frozen['accuracy']:+.4f} accuracy"
            )
        if joint and continual:
            lines.append(
                f"- 与 joint（一次性重训上界）的差距："
                f"{joint['accuracy'] - continual['accuracy']:+.4f} accuracy"
            )
    else:
        lines.append("未评估（run_holdout=False）。")
    lines.append("")
    lines.append("## 4. 说明")
    lines.append("")
    lines.append("- 训练集只增不减，历史批次永不重切；validation 与 holdout 都是冻结集。")
    lines.append(
        "- 这里的“遗忘”用冻结 validation 的轨迹衡量（初始域），"
        "因为归档里没有每批独立的 probe 集。"
    )
    lines.append("- frozen = 只训第一批；continual = 逐批 champion/challenger；joint = 全部批次一次性重训。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
