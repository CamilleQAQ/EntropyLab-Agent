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

from ..agent.cost import CostModel
from ..io.dataset import DatasetSpec, load_dataset_records
from ..training.pipeline import (
    PreparedDataset,
    evaluate_partition,
    extract_features,
    train_dataset,
)
from .decision import Decision, decide
from .plan import BatchPlan
from .selection import _feature_key, features_for, select_and_train


@dataclass
class Champion:
    batch_id: int
    model_id: str
    model_directory: Path
    model: object
    validation_macro_f1: float
    validation_accuracy: float
    # The feature/classifier spec this champion was trained with.  With per-round
    # selection the champion's feature space can differ from the round's anchor.
    spec: object | None = None


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
    for split_name, mask in masks.items():
        if split_name == "train":
            continue
        for index in np.flatnonzero(mask):
            names[int(index)] = split_name
    return names


def _decision(
    batch_id: int,
    challenger: dict,
    champion: Champion | None,
    promoted: bool,
    reason: str,
    retention_drop: float | None = None,
    gain_new: float | None = None,
    probe_scores: dict | None = None,
) -> dict:
    return {
        "batch_id": batch_id,
        "challenger": {
            "model_id": challenger["model_id"],
            "validation_macro_f1": None
            if challenger.get("val_macro_f1") is None
            else round(float(challenger["val_macro_f1"]), 6),
            "validation_accuracy": None
            if challenger.get("val_accuracy") is None
            else round(float(challenger["val_accuracy"]), 6),
            "training_records": int(challenger.get("training_records") or 0),
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
        "retention_drop": None if retention_drop is None else round(float(retention_drop), 6),
        "gain_new": None if gain_new is None else round(float(gain_new), 6),
        "probe_scores": probe_scores,
    }


def run_continual(
    spec: DatasetSpec,
    data_root: Path,
    plan: BatchPlan,
    output_dir: Path,
    *,
    model_prefix: str = "continual",
    run_holdout: bool = True,
    criterion: str = "margin",
    guard_tolerance: float = 0.02,
    min_gain: float = 0.0,
    features_bundle: tuple | None = None,
    select: bool = False,
    selection_budget: float = 12.0,
    pool=None,
    cost_model: CostModel | None = None,
) -> dict:
    if plan.dataset_id != spec.dataset_id:
        raise ValueError(
            f"plan is for {plan.dataset_id!r} but the spec is {spec.dataset_id!r}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if features_bundle is None:
        records = load_dataset_records(spec, data_root)
        plan_info = plan.validate(records)
        records, features, labels, dimension = extract_features(
            spec, data_root, records=records
        )
    else:
        records, features, labels, dimension = features_bundle
        plan_info = plan.validate(records)
    labels = np.asarray(labels, dtype=np.int64)
    healthy = spec.labels.healthy_class_id

    decisions: list[dict] = []
    champion: Champion | None = None
    frozen: Champion | None = None
    trajectory: list[dict] = []

    # Probe sets are never trained on: the reference-domain validation set plus
    # every batch's own probes.  They are what makes the guardrail identifiable
    # even when validation accuracy saturates.
    groups_array = np.asarray([record.group for record in records], dtype=object)
    probe_sets: dict[str, np.ndarray] = {
        "reference": np.isin(groups_array, np.asarray(plan.validation_groups, dtype=object))
    }
    for probe_batch in plan.batches:
        if probe_batch.probe_groups:
            probe_sets[f"batch-{probe_batch.batch_id}"] = np.isin(
                groups_array, np.asarray(probe_batch.probe_groups, dtype=object)
            )

    feature_cache: dict = {}
    feature_cache[_feature_key(spec)] = (features, labels, dimension)
    selection_pool = tuple(pool) if pool is not None else ()
    selection_cost_model = cost_model if cost_model is not None else CostModel()
    windows_per_unit = len(records) * spec.window.windows_per_record

    for batch_index in range(plan.batch_count):
        masks = plan.masks(records, batch_index)
        masks.update(probe_sets)
        training_records = int(masks["train"].sum())

        if select:
            selection = select_and_train(
                spec=spec,
                data_root=Path(data_root),
                records=records,
                labels=labels,
                masks=masks,
                probe_names=tuple(probe_sets),
                champion=champion,
                pool=selection_pool,
                budget_seconds=selection_budget,
                cost_model=selection_cost_model,
                windows=windows_per_unit,
                model_directory=output_dir / "candidates",
                model_prefix=model_prefix,
                batch_index=batch_index,
                healthy=healthy,
                guard_tolerance=guard_tolerance,
                min_gain=min_gain,
                margin=plan.margin,
                feature_cache=feature_cache,
            )
            challenger = {
                "model_id": selection.model_id,
                "val_macro_f1": selection.validation_macro_f1,
                "val_accuracy": selection.validation_accuracy,
                "training_records": training_records,
            }
            selected = selection.selected
            retention_drop = None if selected is None else selected.retention_drop
            gain_new = None if selected is None else selected.gain_new
            probe_scores = {
                "candidates": [outcome.__dict__ for outcome in selection.outcomes],
                "spent_seconds": round(selection.spent_seconds, 3),
                "skipped": selection.skipped,
            }
            promoted = selection.model is not None
            if promoted:
                reason = selected.reason if selected is not None else "committed"
            else:
                reason = "no paid candidate passed the guardrail"
            new_spec = selection.spec
            new_model = selection.model
            new_directory = selection.model_directory
        else:
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
                "training_records": training_records,
            }
            retention_drop = None
            gain_new = None
            probe_scores = None
            if champion is None:
                decision = Decision(True, "initial batch: the first model becomes the champion")
            else:
                if criterion == "guardrail":
                    champion_scores = {
                        name: float(
                            evaluate_partition(
                                prepared, champion.model, name, healthy_class_id=healthy
                            )[0]["multiclass_accuracy"]
                        )
                        for name in probe_sets
                    }
                    challenger_scores = {
                        name: float(
                            evaluate_partition(
                                prepared, result.artifact.model, name, healthy_class_id=healthy
                            )[0]["multiclass_accuracy"]
                        )
                        for name in probe_sets
                    }
                    probe_scores = {
                        "champion": {k: round(v, 4) for k, v in champion_scores.items()},
                        "challenger": {k: round(v, 4) for k, v in challenger_scores.items()},
                    }
                    current = f"batch-{batch_index}"
                    if current in probe_sets:
                        gain_new = challenger_scores[current] - champion_scores[current]
                    else:
                        gain_new = challenger["val_accuracy"] - champion.validation_accuracy
                    previous = [name for name in probe_sets if name != current]
                    retention_drop = (
                        max(champion_scores[name] - challenger_scores[name] for name in previous)
                        if previous
                        else 0.0
                    )
                decision = decide(
                    criterion=criterion,
                    champion_validation_macro_f1=champion.validation_macro_f1,
                    challenger_validation_macro_f1=challenger["val_macro_f1"],
                    margin=plan.margin,
                    retention_drop=retention_drop,
                    gain_new=gain_new,
                    guard_tolerance=guard_tolerance,
                    min_gain=min_gain,
                )
            promoted, reason = decision.promoted, decision.reason
            retention_drop = decision.retention_drop
            gain_new = decision.gain_new
            new_spec = spec
            new_model = result.artifact.model
            new_directory = output_dir / "challengers" / f"batch-{batch_index}"

        decisions.append(
            _decision(
                batch_index,
                challenger,
                champion,
                promoted,
                reason,
                retention_drop,
                gain_new,
                probe_scores,
            )
        )

        if promoted and new_model is not None:
            champion = Champion(
                batch_id=batch_index,
                model_id=challenger["model_id"],
                model_directory=new_directory,
                model=new_model,
                validation_macro_f1=challenger["val_macro_f1"],
                validation_accuracy=challenger["val_accuracy"],
                spec=new_spec,
            )
            if frozen is None:
                frozen = champion
        trajectory.append(
            {
                "batch_id": batch_index,
                "note": plan.batches[batch_index].note,
                "training_records": challenger["training_records"],
                "challenger_macro_f1": challenger["val_macro_f1"],
                "champion_macro_f1": champion.validation_macro_f1 if champion else None,
                "retention_drop": None
                if retention_drop is None
                else round(float(retention_drop), 6),
                "gain_new": None if gain_new is None else round(float(gain_new), 6),
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
    joint_spec = (
        champion.spec
        if (select and champion is not None and champion.spec is not None)
        else spec
    )
    joint_features, joint_labels, joint_dimension = features_for(
        joint_spec, data_root, records, feature_cache
    )
    joint_prepared = _prepared(
        joint_spec, records, joint_features, joint_labels, joint_dimension, joint_masks
    )
    joint_result = train_dataset(
        joint_spec,
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
        spec=joint_spec,
    )

    holdout = None
    if run_holdout:
        # One test phase, three already-frozen models.  Per-record predictions are
        # kept so paired comparisons (McNemar, paired bootstrap) are possible.
        test_indices = np.flatnonzero(joint_masks["test"])
        holdout = {}
        for name, arm in (("frozen", frozen), ("continual", champion), ("joint", joint)):
            if arm is None or arm.model is None:
                continue
            arm_spec = arm.spec if getattr(arm, "spec", None) is not None else spec
            arm_features, arm_labels, arm_dimension = features_for(
                arm_spec, data_root, records, feature_cache
            )
            prepared_arm = _prepared(
                arm_spec, records, arm_features, arm_labels, arm_dimension, joint_masks
            )
            metrics, record_predictions, _, _ = evaluate_partition(
                prepared_arm, arm.model, "test", healthy_class_id=healthy
            )
            holdout[name] = {
                "accuracy": float(metrics["multiclass_accuracy"]),
                "macro_f1": float(metrics["multiclass_macro_f1"]),
                "records": int(joint_masks["test"].sum()),
                "predictions": [
                    {
                        "record_id": records[int(index)].record_id,
                        "truth": int(labels[int(index)]),
                        "predicted": int(record_predictions[position]),
                    }
                    for position, index in enumerate(test_indices)
                ],
            }

    summary = {
        "dataset_id": spec.dataset_id,
        "configuration": {
            "feature": spec.feature.method,
            "feature_dimension": int(dimension),
            "classifier": spec.classifier.algorithm,
        },
        "plan": plan_info,
        "selection": {
            "enabled": bool(select),
            "budget_seconds": selection_budget if select else None,
            "pool_size": len(selection_pool) if select else None,
        },
        "champion_configuration": None
        if champion is None or getattr(champion, "spec", None) is None
        else {
            "feature": champion.spec.feature.method,
            "classifier": champion.spec.classifier.algorithm,
        },
        "criterion": criterion,
        "guard_tolerance": guard_tolerance,
        "min_gain": min_gain,
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


def _fmt_metric(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


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
            f"{_fmt_metric(row['challenger_macro_f1'])} | "
            f"{_fmt_metric(row['champion_macro_f1'])} | {decision} |"
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
