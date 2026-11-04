"""The 5-round Agent loop: search on validation, open test exactly once.

Arms share one feature cache, so the cost of a run is dominated by the number of
distinct (method, parameters) pairs, not by the number of trials.  Every trial is
written to the SQLite experiment memory, including failures.

Discipline enforced here:

* trials are scored on the validation partition only;
* the champion is selected by validation macro-F1 (tie-break: accuracy, then the
  earlier trial);
* the test partition is featurized and scored once, for the champion only, after
  the search has finished.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..io.dataset import (
    ClassifierConfig,
    DatasetRecord,
    DatasetSpec,
    FeatureConfig,
    load_dataset_records,
    resolve_dataset_split,
)
from ..training.pipeline import (
    PreparedDataset,
    evaluate_partition,
    fit_prepared,
    train_dataset,
)
from .cache import FeatureCache
from .candidates import Candidate, build_pool
from .memory import ExperimentMemory
from .profile import DataProfile, build_profile
from .selector import anchor_candidate, propose


@dataclass
class TrialResult:
    arm: str
    trial_index: int
    round_index: int
    candidate: Candidate
    status: str
    failure: str | None
    val_accuracy: float | None
    val_macro_f1: float | None
    cache_hit: bool
    extract_seconds: float
    fit_seconds: float
    feature_key: str

    def history(self) -> dict:
        return {
            "candidate_id": self.candidate.candidate_id,
            "method": self.candidate.method.method,
            "family": self.candidate.family,
            "classifier": self.candidate.classifier.algorithm,
            "status": self.status,
            "val_accuracy": self.val_accuracy,
            "val_macro_f1": self.val_macro_f1,
        }


@dataclass
class _Context:
    spec: DatasetSpec
    data_root: Path
    records_dev: list[DatasetRecord]
    split_by_group: dict[str, str]
    cache: FeatureCache
    memory: ExperimentMemory
    run_id: str


def _spec_with_candidate(spec: DatasetSpec, candidate: Candidate) -> DatasetSpec:
    return spec.model_copy(
        update={
            "feature": FeatureConfig(
                method=candidate.method.method,
                parameters=dict(candidate.method.parameters),
            ),
            "classifier": ClassifierConfig(
                algorithm=candidate.classifier.algorithm,
                backend=candidate.classifier.backend,
                parameters=dict(candidate.classifier.parameters),
            ),
        }
    )


def _prepared(
    spec: DatasetSpec,
    records: list[DatasetRecord],
    features: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    split_by_group: dict[str, str],
    feature_seconds: float,
) -> PreparedDataset:
    groups = tuple(record.group for record in records)
    split = np.asarray([split_by_group[group] for group in groups])
    masks = {name: split == name for name in ("train", "validation", "test")}
    return PreparedDataset(
        spec=spec,
        records=records,
        features=features,
        labels=labels,
        groups=groups,
        feature_dimension=int(dimension),
        split_by_group=split_by_group,
        masks=masks,
        feature_seconds=feature_seconds,
    )


def _trial(
    context: _Context,
    candidate: Candidate,
    *,
    arm: str,
    trial_index: int,
    round_index: int,
) -> TrialResult:
    spec = _spec_with_candidate(context.spec, candidate)
    cache_hit = False
    extract_seconds = 0.0
    fit_seconds = 0.0
    feature_key = ""
    status = "OK"
    failure: str | None = None
    val_accuracy: float | None = None
    val_macro_f1: float | None = None
    try:
        feature_set = context.cache.get_or_extract(spec, context.data_root, context.records_dev)
        cache_hit = feature_set.hit
        extract_seconds = feature_set.seconds
        feature_key = feature_set.key
        prepared = _prepared(
            spec,
            context.records_dev,
            feature_set.features,
            feature_set.labels,
            feature_set.dimension,
            context.split_by_group,
            feature_set.seconds,
        )
        model, fit_seconds = fit_prepared(prepared, spec)
        metrics, _, _, _ = evaluate_partition(
            prepared,
            model,
            "validation",
            healthy_class_id=context.spec.labels.healthy_class_id,
        )
        val_accuracy = float(metrics["multiclass_accuracy"])
        val_macro_f1 = float(metrics["multiclass_macro_f1"])
    except Exception as error:  # failures are data for the Agent, not crashes
        status = "FAILED"
        failure = f"{type(error).__name__}: {error}"

    result = TrialResult(
        arm=arm,
        trial_index=trial_index,
        round_index=round_index,
        candidate=candidate,
        status=status,
        failure=failure,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        cache_hit=cache_hit,
        extract_seconds=extract_seconds,
        fit_seconds=fit_seconds,
        feature_key=feature_key,
    )
    context.memory.record_trial(
        run_id=context.run_id,
        arm=arm,
        trial_index=trial_index,
        round_index=round_index,
        candidate=candidate.describe(),
        feature_key=feature_key,
        cache_hit=cache_hit,
        status=status,
        failure=failure,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        extract_seconds=extract_seconds,
        fit_seconds=fit_seconds,
    )
    return result


def _agent_arm(
    context: _Context,
    pool: Sequence[Candidate],
    rounds: int,
    per_round: int,
    anchor_id: str | None,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    history: list[dict] = []
    index = 0
    for round_index in range(rounds):
        batch = propose(round_index, history, pool, per_round, anchor_id)
        if not batch:
            break
        for candidate in batch:
            result = _trial(
                context, candidate, arm="agent", trial_index=index, round_index=round_index
            )
            results.append(result)
            history.append(result.history())
            index += 1
    return results


def _random_arm(
    context: _Context, pool: Sequence[Candidate], budget: int, seed: int
) -> list[TrialResult]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(pool))[:budget]
    return [
        _trial(context, pool[int(position)], arm="random", trial_index=index, round_index=0)
        for index, position in enumerate(order)
    ]


def _fixed_arm(
    context: _Context, pool: Sequence[Candidate], budget: int
) -> list[TrialResult]:
    return [
        _trial(context, candidate, arm="fixed", trial_index=index, round_index=0)
        for index, candidate in enumerate(pool[:budget])
    ]


def _exhaustive_arm(context: _Context, pool: Sequence[Candidate]) -> list[TrialResult]:
    return [
        _trial(context, candidate, arm="exhaustive", trial_index=index, round_index=0)
        for index, candidate in enumerate(pool)
    ]


def _champion(results: Sequence[TrialResult]) -> TrialResult | None:
    scored = [
        result
        for result in results
        if result.status == "OK" and result.val_macro_f1 is not None
    ]
    if not scored:
        return None
    return max(
        scored,
        key=lambda result: (
            result.val_macro_f1,
            result.val_accuracy if result.val_accuracy is not None else -1.0,
            -result.trial_index,
        ),
    )


def champion_diagnosis(spec: DatasetSpec, candidate: Candidate):
    """Diagnosis profile whose display name matches the selected model.

    The configured display name usually ends with the configured feature and
    classifier (for example ``"... , TSMEFuDE + BT-LSTSVM"``).  Keeping that text
    on an agent-selected artifact would describe a model that was not shipped, so
    the configured suffix is replaced by the selected combination.  The profile
    ID stays the same because it describes the diagnosis contract, not the
    algorithm that happens to win.
    """
    configured = spec.diagnosis.display_name
    configured_suffix = f"{spec.feature.method} + {spec.classifier.algorithm}"
    base = configured
    if configured.endswith(configured_suffix):
        base = configured[: -len(configured_suffix)].rstrip(" ,;:—-–")
    selected = f"{candidate.method.method} + {candidate.classifier.algorithm}"
    display_name = f"{base} — agent-selected {selected}" if base else f"agent-selected {selected}"
    return spec.diagnosis.model_copy(update={"display_name": display_name})


def champion_model_id(spec: DatasetSpec, candidate: Candidate) -> str:
    """Stable, readable model id that names the selected combination."""
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{candidate.method.method}-{candidate.classifier.algorithm}".lower(),
    ).strip("-")
    return f"{spec.dataset_id.lower()}-agent-{slug}"


def _evaluate_champion(
    context: _Context,
    champion: TrialResult,
    records_all: list[DatasetRecord],
    model_directory: Path,
    model_id: str,
) -> dict:
    spec = _spec_with_candidate(context.spec, champion.candidate).model_copy(
        update={"diagnosis": champion_diagnosis(context.spec, champion.candidate)}
    )
    development = context.cache.get_or_extract(spec, context.data_root, context.records_dev)
    test_records = [
        record
        for record in records_all
        if context.split_by_group[record.group] == "test"
    ]
    if not test_records:
        raise ValueError("champion test evaluation requires a non-empty test partition")
    # The only time the test partition is featurized in this run.
    test_features = context.cache.get_or_extract(spec, context.data_root, test_records)
    rows = {
        record_id: development.features[index]
        for index, record_id in enumerate(development.record_ids)
    }
    rows.update(
        {
            record_id: test_features.features[index]
            for index, record_id in enumerate(test_features.record_ids)
        }
    )
    features = np.stack([rows[record.record_id] for record in records_all], axis=0)
    labels = np.asarray([record.label for record in records_all], dtype=np.int64)
    prepared = _prepared(
        spec,
        records_all,
        features,
        labels,
        development.dimension,
        context.split_by_group,
        development.seconds + test_features.seconds,
    )
    return train_dataset(
        spec,
        context.data_root,
        model_directory,
        model_id=model_id,
        prepared=prepared,
        evaluate_test=True,
    ).test_result


def _write_trials_csv(path: Path, results: Sequence[TrialResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm", "trial", "round", "method", "family", "classifier",
                "status", "val_accuracy", "val_macro_f1", "cache_hit",
                "extract_seconds", "fit_seconds", "failure",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.arm, result.trial_index, result.round_index,
                    result.candidate.method.method, result.candidate.family,
                    result.candidate.classifier.algorithm, result.status,
                    result.val_accuracy, result.val_macro_f1, int(result.cache_hit),
                    round(result.extract_seconds, 3), round(result.fit_seconds, 4),
                    result.failure or "",
                ]
            )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _write_report(
    path: Path,
    *,
    run_id: str,
    profile: DataProfile,
    budget: dict,
    results_by_arm: dict[str, list[TrialResult]],
    champion: TrialResult | None,
    test_metrics: dict | None,
    model_directory: Path,
    model_id: str | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Agent v1 运行报告")
    lines.append("")
    lines.append(f"- run_id：`{run_id}`")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 数据集：`{profile.dataset_id}`（{profile.class_count} 类，"
                 f"窗口 {profile.window_size}，split = {profile.split_mode}）")
    lines.append(
        f"- 预算：{budget['rounds']} 轮 × {budget['per_round']} 个 = "
        f"{budget['rounds'] * budget['per_round']} 个 trial（穷举臂 {budget['pool_size']} 个）"
    )
    lines.append("")
    lines.append("## 1. 数据概况")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 类别 | {', '.join(profile.class_names)} |")
    lines.append(f"| 分区记录数 | {profile.records_per_split} |")
    lines.append(f"| 组数 | {profile.group_count} |")
    lines.append(f"| 窗长 / 每记录窗口数 | {profile.window_size} / {profile.windows_per_record} |")
    lines.append("")

    lines.append("## 2. 各臂对比（只看 validation）")
    lines.append("")
    lines.append("| 臂 | trial 数 | 最佳 Macro-F1 | 对应 Accuracy | 缓存命中 |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm, results in results_by_arm.items():
        best = _champion(results)
        hits = sum(1 for result in results if result.cache_hit)
        lines.append(
            f"| {arm} | {len(results)} | {_fmt(best.val_macro_f1 if best else None)} | "
            f"{_fmt(best.val_accuracy if best else None)} | {hits}/{len(results)} |"
        )
    lines.append("")

    agent_results = results_by_arm.get("agent", [])
    if agent_results:
        lines.append("## 3. Agent 每轮选择")
        lines.append("")
        lines.append("| 轮次 | trial | 方法 | 分类器 | val Macro-F1 | 特征耗时 | 状态 |")
        lines.append("|---:|---:|---|---|---:|---:|---|")
        for result in agent_results:
            lines.append(
                f"| {result.round_index} | {result.trial_index} | {result.candidate.method.method} | "
                f"{result.candidate.classifier.algorithm} | {_fmt(result.val_macro_f1)} | "
                f"{result.extract_seconds:.1f}s | {result.status} |"
            )
        lines.append("")

    exhaustive = results_by_arm.get("exhaustive", [])
    if exhaustive:
        ranked = sorted(
            [result for result in exhaustive if result.val_macro_f1 is not None],
            key=lambda result: (-result.val_macro_f1, -result.val_accuracy),
        )[:5]
        lines.append("## 4. 穷举上界（前 5）")
        lines.append("")
        lines.append("| 排名 | 方法 | 分类器 | val Macro-F1 | val Accuracy |")
        lines.append("|---:|---|---|---:|---:|")
        for rank, result in enumerate(ranked, start=1):
            lines.append(
                f"| {rank} | {result.candidate.method.method} | "
                f"{result.candidate.classifier.algorithm} | {_fmt(result.val_macro_f1)} | "
                f"{_fmt(result.val_accuracy)} |"
            )
        lines.append("")

    lines.append("## 5. 冠军与唯一一次 test")
    lines.append("")
    if champion is None or test_metrics is None:
        lines.append("没有可用冠军（全部 trial 失败）。")
    else:
        lines.append(f"- 冠军：`{champion.candidate.label}`（agent 第 {champion.round_index} 轮，"
                     f"trial {champion.trial_index}）")
        if model_id:
            lines.append(f"- model_id：`{model_id}`")
        lines.append(f"- validation：accuracy = {_fmt(champion.val_accuracy)}，"
                     f"macro-F1 = {_fmt(champion.val_macro_f1)}")
        lines.append(f"- 模型：`{model_directory}`")
        lines.append("")
        lines.append("| 分区 | Accuracy | Macro-F1 |")
        lines.append("|---|---:|---:|")
        for name in ("validation", "test"):
            metrics = test_metrics[name]
            lines.append(
                f"| {name} | {_fmt(metrics['multiclass_accuracy'])} | "
                f"{_fmt(metrics['multiclass_macro_f1'])} |"
            )
        lines.append("")
        lines.append("test 只在上述冠军上计算一次；其余臂只报告 validation。")
    lines.append("")

    failures = [
        result
        for results in results_by_arm.values()
        for result in results
        if result.status != "OK"
    ]
    lines.append("## 6. 失败 trial")
    lines.append("")
    if not failures:
        lines.append("无。")
    else:
        lines.append("| 臂 | 方法 | 分类器 | 失败原因 |")
        lines.append("|---|---|---|---|")
        for result in failures:
            lines.append(
                f"| {result.arm} | {result.candidate.method.method} | "
                f"{result.candidate.classifier.algorithm} | {result.failure} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_agent(
    spec: DatasetSpec,
    data_root: Path,
    output_dir: Path,
    *,
    rounds: int = 5,
    per_round: int = 3,
    seed: int = 20261104,
    arms: Sequence[str] = ("agent", "random", "fixed", "exhaustive"),
    pool: Sequence[Candidate] | None = None,
    cache_root: Path | None = None,
    memory_path: Path | None = None,
    model_id: str | None = None,
    run_test: bool = True,
    run_id: str | None = None,
) -> dict:
    """Run the Agent and its same-budget controls; return the summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = tuple(pool) if pool is not None else build_pool()
    known_arms = {"agent", "random", "fixed", "exhaustive"}
    unknown = set(arms) - known_arms
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    budget = rounds * per_round
    if budget > len(pool):
        raise ValueError(
            f"budget {budget} exceeds the candidate pool size {len(pool)}"
        )

    records_all = load_dataset_records(spec, data_root)
    split_by_group = resolve_dataset_split(spec, records_all)
    records_dev = [
        record for record in records_all if split_by_group[record.group] != "test"
    ]
    profile = build_profile(spec, records_all, split_by_group)

    run_id = run_id or (
        f"agent-v1-{spec.dataset_id.lower().replace('_', '-')}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    memory = ExperimentMemory(memory_path or output_dir / "memory.sqlite")
    anchor = anchor_candidate(pool)
    try:
        memory.start_run(
            run_id,
            spec.dataset_id,
            {"profile": profile.as_dict(), "pool_size": len(pool),
             "anchor": anchor.candidate_id if anchor else None},
            {"rounds": rounds, "per_round": per_round, "seed": seed,
             "pool_size": len(pool), "arms": list(arms)},
        )
        context = _Context(
            spec=spec,
            data_root=Path(data_root),
            records_dev=records_dev,
            split_by_group=split_by_group,
            cache=FeatureCache(
                Path(cache_root) if cache_root is not None else output_dir / "feature_cache"
            ),
            memory=memory,
            run_id=run_id,
        )

        results_by_arm: dict[str, list[TrialResult]] = {}
        if "agent" in arms:
            results_by_arm["agent"] = _agent_arm(
                context, pool, rounds, per_round, anchor.candidate_id if anchor else None
            )
        if "random" in arms:
            results_by_arm["random"] = _random_arm(context, pool, budget, seed)
        if "fixed" in arms:
            results_by_arm["fixed"] = _fixed_arm(context, pool, budget)
        if "exhaustive" in arms:
            results_by_arm["exhaustive"] = _exhaustive_arm(context, pool)

        all_results = [result for results in results_by_arm.values() for result in results]
        _write_trials_csv(output_dir / "trials.csv", all_results)

        champion = _champion(results_by_arm.get("agent", []))
        test_metrics: dict | None = None
        model_directory = output_dir / "champion_model"
        resolved_model_id = (
            model_id
            or (
                champion_model_id(spec, champion.candidate)
                if champion is not None
                else None
            )
        )
        if run_test and champion is not None:
            test_metrics = _evaluate_champion(
                context,
                champion,
                records_all,
                model_directory,
                resolved_model_id,
            )
        memory.finish_run(
            run_id,
            champion.candidate.describe() if champion is not None else None,
            test_metrics,
        )
        _write_report(
            output_dir / "report.md",
            run_id=run_id,
            profile=profile,
            budget={
                "rounds": rounds, "per_round": per_round,
                "pool_size": len(pool),
            },
            results_by_arm=results_by_arm,
            champion=champion,
            test_metrics=test_metrics,
            model_directory=model_directory,
            model_id=resolved_model_id,
        )
        arm_summary = {}
        for arm, results in results_by_arm.items():
            best = _champion(results)
            arm_summary[arm] = {
                "trials": len(results),
                "best_val_macro_f1": best.val_macro_f1 if best else None,
                "best_val_accuracy": best.val_accuracy if best else None,
                "best_candidate": best.candidate.label if best else None,
            }
        summary = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "profile": profile.as_dict(),
            "budget": {"rounds": rounds, "per_round": per_round, "pool_size": len(pool)},
            "arms": arm_summary,
            "champion": champion.candidate.describe() if champion is not None else None,
            "champion_validation": (
                {"accuracy": champion.val_accuracy, "macro_f1": champion.val_macro_f1}
                if champion is not None
                else None
            ),
            "test": test_metrics,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        memory.close()
