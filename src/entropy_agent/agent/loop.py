"""The 5-round Agent loop (v1.1): repeated validation, multi-seed, one test.

What changed from v1:

* features are cached per record, so a new split seed or a new classifier pays
  nothing for extraction;
* every trial is scored on several group-isolated validation splits and reports
  mean and standard deviation instead of a single fragile number;
* the same arms are repeated over several split seeds and the report aggregates
  them, so "agent beats random" can be stated with a spread;
* each proposal carries the selector's reason into the memory and the report.

Discipline is unchanged: selection reads validation only, and the test partition
is scored exactly once, for the champion, after every arm has finished.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    extract_features,
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
    seed: int
    trial_index: int
    round_index: int
    candidate: Candidate
    reason: str
    status: str
    failure: str | None
    val_accuracy: float | None
    val_macro_f1: float | None
    val_accuracy_std: float | None
    val_macro_f1_std: float | None
    cache_hit: bool
    extracted_records: int
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
    records: list[DatasetRecord]
    labels: np.ndarray
    cache: FeatureCache
    memory: ExperimentMemory
    run_id: str
    seed: int
    splits: list[dict[str, str]] = field(default_factory=list)
    extracted_methods: set[str] = field(default_factory=set)

    @property
    def primary_split(self) -> dict[str, str]:
        return self.splits[0]


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


def champion_diagnosis(spec: DatasetSpec, candidate: Candidate):
    """Diagnosis profile whose display name matches the selected model."""
    configured = spec.diagnosis.display_name
    configured_suffix = f"{spec.feature.method} + {spec.classifier.algorithm}"
    base = configured
    if configured.endswith(configured_suffix):
        base = configured[: -len(configured_suffix)].rstrip(" ,;:—-–")
    selected = f"{candidate.method.method} + {candidate.classifier.algorithm}"
    display_name = f"{base} — agent-selected {selected}" if base else f"agent-selected {selected}"
    return spec.diagnosis.model_copy(update={"display_name": display_name})


def champion_model_id(spec: DatasetSpec, candidate: Candidate) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{candidate.method.method}-{candidate.classifier.algorithm}".lower(),
    ).strip("-")
    return f"{spec.dataset_id.lower()}-agent-{slug}"


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
    reason: str,
) -> TrialResult:
    spec = _spec_with_candidate(context.spec, candidate)
    status = "OK"
    failure: str | None = None
    cache_hit = False
    extracted_records = 0
    extract_seconds = 0.0
    fit_seconds = 0.0
    feature_key = ""
    f1_values: list[float] = []
    accuracy_values: list[float] = []
    try:
        feature_set = context.cache.get_or_extract(spec, context.data_root, context.records)
        cache_hit = feature_set.hit
        extracted_records = feature_set.extracted_records
        extract_seconds = feature_set.seconds
        feature_key = feature_set.key
        # Selection must not depend on whether the cache happened to be warm.
        context.extracted_methods.add(candidate.method.method)
        for split_by_group in context.splits:
            prepared = _prepared(
                spec,
                context.records,
                feature_set.features,
                context.labels,
                feature_set.dimension,
                split_by_group,
                feature_set.seconds,
            )
            model, seconds = fit_prepared(prepared, spec)
            fit_seconds += seconds
            metrics, _, _, _ = evaluate_partition(
                prepared,
                model,
                "validation",
                healthy_class_id=context.spec.labels.healthy_class_id,
            )
            f1_values.append(float(metrics["multiclass_macro_f1"]))
            accuracy_values.append(float(metrics["multiclass_accuracy"]))
    except Exception as error:  # failures are data for the Agent, not crashes
        status = "FAILED"
        failure = f"{type(error).__name__}: {error}"

    if f1_values:
        val_macro_f1 = float(np.mean(f1_values))
        val_macro_f1_std = float(np.std(f1_values))
        val_accuracy = float(np.mean(accuracy_values))
        val_accuracy_std = float(np.std(accuracy_values))
    else:
        val_macro_f1 = val_macro_f1_std = val_accuracy = val_accuracy_std = None

    result = TrialResult(
        arm=arm,
        seed=context.seed,
        trial_index=trial_index,
        round_index=round_index,
        candidate=candidate,
        reason=reason,
        status=status,
        failure=failure,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        val_accuracy_std=val_accuracy_std,
        val_macro_f1_std=val_macro_f1_std,
        cache_hit=cache_hit,
        extracted_records=extracted_records,
        extract_seconds=extract_seconds,
        fit_seconds=fit_seconds,
        feature_key=feature_key,
    )
    context.memory.record_trial(
        run_id=context.run_id,
        arm=arm,
        seed=context.seed,
        trial_index=trial_index,
        round_index=round_index,
        candidate=candidate.describe(),
        feature_key=feature_key,
        cache_hit=cache_hit,
        extracted_records=extracted_records,
        status=status,
        failure=failure,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        val_accuracy_std=val_accuracy_std,
        val_macro_f1_std=val_macro_f1_std,
        reason=reason,
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
        batch = propose(
            round_index,
            history,
            pool,
            per_round,
            anchor_id,
            context.extracted_methods,
        )
        if not batch:
            break
        for proposal in batch:
            result = _trial(
                context,
                proposal.candidate,
                arm="agent",
                trial_index=index,
                round_index=round_index,
                reason=proposal.reason,
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
        _trial(
            context,
            pool[int(position)],
            arm="random",
            trial_index=index,
            round_index=0,
            reason="random sample",
        )
        for index, position in enumerate(order)
    ]


def _fixed_arm(
    context: _Context, pool: Sequence[Candidate], budget: int
) -> list[TrialResult]:
    return [
        _trial(
            context,
            candidate,
            arm="fixed",
            trial_index=index,
            round_index=0,
            reason="fixed pool order",
        )
        for index, candidate in enumerate(pool[:budget])
    ]


def _exhaustive_arm(context: _Context, pool: Sequence[Candidate]) -> list[TrialResult]:
    return [
        _trial(
            context,
            candidate,
            arm="exhaustive",
            trial_index=index,
            round_index=0,
            reason="exhaustive",
        )
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
            -(result.val_macro_f1_std or 0.0),
            result.val_accuracy if result.val_accuracy is not None else -1.0,
            -result.trial_index,
        ),
    )


def _evaluate_champion(
    context: _Context,
    champion: TrialResult,
    model_directory: Path,
    model_id: str,
) -> dict:
    spec = _spec_with_candidate(context.spec, champion.candidate).model_copy(
        update={"diagnosis": champion_diagnosis(context.spec, champion.candidate)}
    )
    feature_set = context.cache.get_or_extract(spec, context.data_root, context.records)
    prepared = _prepared(
        spec,
        context.records,
        feature_set.features,
        context.labels,
        feature_set.dimension,
        context.primary_split,
        feature_set.seconds,
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
                "arm", "seed", "trial", "round", "method", "family", "classifier",
                "status", "val_macro_f1", "val_macro_f1_std", "val_accuracy",
                "cache_hit", "extracted_records", "extract_seconds", "fit_seconds",
                "reason", "failure",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.arm, result.seed, result.trial_index, result.round_index,
                    result.candidate.method.method, result.candidate.family,
                    result.candidate.classifier.algorithm, result.status,
                    result.val_macro_f1, result.val_macro_f1_std, result.val_accuracy,
                    int(result.cache_hit), result.extracted_records,
                    round(result.extract_seconds, 3), round(result.fit_seconds, 4),
                    result.reason, result.failure or "",
                ]
            )


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _aggregate_seeds(
    results_by_seed: dict[int, dict[str, list[TrialResult]]],
) -> dict[str, dict]:
    aggregate: dict[str, dict] = {}
    for seed, arms in results_by_seed.items():
        for arm, results in arms.items():
            best = _champion(results)
            entry = aggregate.setdefault(arm, {"per_seed": {}, "values": []})
            entry["per_seed"][str(seed)] = {
                "best_val_macro_f1": best.val_macro_f1 if best else None,
                "best_candidate": best.candidate.label if best else None,
            }
            if best is not None and best.val_macro_f1 is not None:
                entry["values"].append(best.val_macro_f1)
    for entry in aggregate.values():
        values = entry.pop("values")
        entry["mean"] = float(np.mean(values)) if values else None
        entry["std"] = float(np.std(values)) if values else None
        entry["wins"] = 0
    seeds = sorted(results_by_seed)
    for seed in seeds:
        values = {
            arm: aggregate[arm]["per_seed"].get(str(seed), {}).get("best_val_macro_f1")
            for arm in aggregate
        }
        finite = [value for value in values.values() if value is not None]
        if not finite:
            continue
        best_value = max(finite)
        # Ties are shared wins, not a win for whichever arm happens to be first.
        for arm, value in values.items():
            if value is not None and abs(value - best_value) <= 1e-12:
                aggregate[arm]["wins"] += 1
    return aggregate


def _write_report(
    path: Path,
    *,
    run_id: str,
    profile: DataProfile,
    budget: dict,
    results_by_seed: dict[int, dict[str, list[TrialResult]]],
    aggregate: dict[str, dict],
    champion: TrialResult | None,
    test_metrics: dict | None,
    model_directory: Path,
    model_id: str | None,
) -> None:
    lines: list[str] = []
    lines.append("# Agent v1.1 运行报告")
    lines.append("")
    lines.append(f"- run_id：`{run_id}`")
    lines.append(f"- 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append(
        f"- 数据集：`{profile.dataset_id}`（{profile.class_count} 类，窗口 {profile.window_size}）"
    )
    lines.append(
        f"- 预算：{budget['rounds']} 轮 × {budget['per_round']} = "
        f"{budget['rounds'] * budget['per_round']} 个 trial；"
        f"seed 数 {budget['seeds']}；每 trial {budget['validation_repeats']} 次切分"
    )
    lines.append("")
    lines.append("## 1. 数据概况")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 类别 | {', '.join(profile.class_names)} |")
    lines.append(f"| 分区记录数 | {profile.records_per_split} |")
    lines.append(f"| 组数 | {profile.group_count} |")
    lines.append("")

    lines.append("## 2. 各臂在多个 seed 上的表现（只看 validation）")
    lines.append("")
    lines.append("| 臂 | 均值 Macro-F1 | 标准差 | 胜场 | trial/seed |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm, stats in aggregate.items():
        lines.append(
            f"| {arm} | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
            f"{stats['wins']} | {len(results_by_seed[sorted(results_by_seed)[0]][arm])} |"
        )
    lines.append("")
    lines.append("每个 seed 的最佳 validation：")
    lines.append("")
    arms = list(aggregate)
    header = "| seed | " + " | ".join(arms) + " |"
    lines.append(header)
    lines.append("|---:|" + "---:|" * len(arms))
    for seed in sorted(results_by_seed):
        row = [f"| {seed} "]
        for arm in arms:
            value = aggregate[arm]["per_seed"].get(str(seed), {}).get("best_val_macro_f1")
            row.append(f"| {_fmt(value)} ")
        row.append("|")
        lines.append("".join(row))
    lines.append("")

    primary = sorted(results_by_seed)[0]
    agent_results = results_by_seed[primary]["agent"]
    lines.append(f"## 3. Agent 每轮选择（seed {primary}）")
    lines.append("")
    lines.append("| 轮 | trial | 方法 | 分类器 | val Macro-F1 ± std | 提取记录 | 理由 |")
    lines.append("|---:|---:|---|---|---:|---:|---|")
    for result in agent_results:
        std = "—" if result.val_macro_f1_std is None else f"{result.val_macro_f1_std:.4f}"
        lines.append(
            f"| {result.round_index} | {result.trial_index} | {result.candidate.method.method} | "
            f"{result.candidate.classifier.algorithm} | {_fmt(result.val_macro_f1)} ± {std} | "
            f"{result.extracted_records} | {result.reason} |"
        )
    lines.append("")

    exhaustive = results_by_seed[primary].get("exhaustive", [])
    if exhaustive:
        ranked = sorted(
            [result for result in exhaustive if result.val_macro_f1 is not None],
            key=lambda result: (-result.val_macro_f1, -(result.val_macro_f1_std or 0.0)),
        )[:5]
        lines.append(f"## 4. 穷举上界前 5（seed {primary}）")
        lines.append("")
        lines.append("| 排名 | 方法 | 分类器 | val Macro-F1 ± std |")
        lines.append("|---:|---|---|---:|")
        for rank, result in enumerate(ranked, start=1):
            std = "—" if result.val_macro_f1_std is None else f"{result.val_macro_f1_std:.4f}"
            lines.append(
                f"| {rank} | {result.candidate.method.method} | "
                f"{result.candidate.classifier.algorithm} | {_fmt(result.val_macro_f1)} ± {std} |"
            )
        lines.append("")

    lines.append("## 5. 冠军与唯一一次 test")
    lines.append("")
    if champion is None or test_metrics is None:
        lines.append("没有可用冠军（全部 trial 失败）。")
    else:
        lines.append(f"- 冠军：`{champion.candidate.label}`（seed {champion.seed}，"
                     f"第 {champion.round_index} 轮，trial {champion.trial_index}）")
        if model_id:
            lines.append(f"- model_id：`{model_id}`")
        lines.append(
            f"- validation（{budget['validation_repeats']} 次切分均值）："
            f"{_fmt(champion.val_macro_f1)} ± "
            f"{_fmt(champion.val_macro_f1_std)}"
        )
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
        lines.append("test 只在上述冠军上计算一次；其余臂与其余 seed 只报告 validation。")
    lines.append("")

    failures = [
        result
        for arms_by_seed in results_by_seed.values()
        for results in arms_by_seed.values()
        for result in results
        if result.status != "OK"
    ]
    lines.append("## 6. 失败 trial")
    lines.append("")
    if not failures:
        lines.append("无。")
    else:
        lines.append("| seed | 臂 | 方法 | 分类器 | 失败原因 |")
        lines.append("|---:|---|---|---|---|")
        for result in failures:
            lines.append(
                f"| {result.seed} | {result.arm} | {result.candidate.method.method} | "
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
    seeds: Sequence[int] = (20261104,),
    validation_repeats: int = 3,
    arms: Sequence[str] = ("agent", "random", "fixed", "exhaustive"),
    pool: Sequence[Candidate] | None = None,
    cache_root: Path | None = None,
    memory_path: Path | None = None,
    model_id: str | None = None,
    run_test: bool = True,
    run_id: str | None = None,
) -> dict:
    """Run the Agent and its same-budget controls across split seeds."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = tuple(pool) if pool is not None else build_pool()
    known_arms = {"agent", "random", "fixed", "exhaustive"}
    unknown = set(arms) - known_arms
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    budget = rounds * per_round
    if budget > len(pool):
        raise ValueError(f"budget {budget} exceeds the candidate pool size {len(pool)}")
    if validation_repeats < 1:
        raise ValueError("validation_repeats must be positive")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")

    records = load_dataset_records(spec, data_root)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    profile = build_profile(spec, records, resolve_dataset_split(spec, records))

    run_id = run_id or (
        f"agent-v11-{spec.dataset_id.lower().replace('_', '-')}-"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    memory = ExperimentMemory(memory_path or output_dir / "memory.sqlite")
    cache = FeatureCache(
        Path(cache_root) if cache_root is not None else output_dir / "feature_cache"
    )
    anchor = anchor_candidate(pool)
    try:
        memory.start_run(
            run_id,
            spec.dataset_id,
            {"profile": profile.as_dict(), "pool_size": len(pool),
             "anchor": anchor.candidate_id if anchor else None},
            {"rounds": rounds, "per_round": per_round, "pool_size": len(pool),
             "arms": list(arms), "seeds": list(seeds),
             "validation_repeats": validation_repeats},
        )

        # Features are extracted lazily per entropy method and cached per record,
        # so a new split seed or a new classifier costs nothing to re-featurize.
        results_by_seed: dict[int, dict[str, list[TrialResult]]] = {}
        for seed in seeds:
            splits = [
                resolve_dataset_split(
                    spec.model_copy(
                        update={"split": spec.split.model_copy(update={"seed": seed + repeat})}
                    ),
                    records,
                )
                for repeat in range(validation_repeats)
            ]
            context = _Context(
                spec=spec,
                data_root=Path(data_root),
                records=records,
                labels=labels,
                cache=cache,
                memory=memory,
                run_id=run_id,
                seed=seed,
                splits=splits,
            )
            arms_results: dict[str, list[TrialResult]] = {}
            if "agent" in arms:
                arms_results["agent"] = _agent_arm(
                    context, pool, rounds, per_round,
                    anchor.candidate_id if anchor else None,
                )
            if "random" in arms:
                arms_results["random"] = _random_arm(context, pool, budget, seed)
            if "fixed" in arms:
                arms_results["fixed"] = _fixed_arm(context, pool, budget)
            if "exhaustive" in arms:
                arms_results["exhaustive"] = _exhaustive_arm(context, pool)
            results_by_seed[seed] = arms_results

        all_results = [
            result
            for arms_by_seed in results_by_seed.values()
            for results in arms_by_seed.values()
            for result in results
        ]
        _write_trials_csv(output_dir / "trials.csv", all_results)
        aggregate = _aggregate_seeds(results_by_seed)

        champion = _champion(
            [
                result
                for results in results_by_seed.values()
                for result in results.get("agent", [])
            ]
        )
        test_metrics: dict | None = None
        model_directory = output_dir / "champion_model"
        resolved_model_id = (
            model_id
            or (champion_model_id(spec, champion.candidate) if champion is not None else None)
        )
        if run_test and champion is not None:
            test_context = _Context(
                spec=spec,
                data_root=Path(data_root),
                records=records,
                labels=labels,
                cache=cache,
                memory=memory,
                run_id=run_id,
                seed=champion.seed,
                splits=[
                    resolve_dataset_split(
                        spec.model_copy(
                            update={
                                "split": spec.split.model_copy(
                                    update={"seed": champion.seed}
                                )
                            }
                        ),
                        records,
                    )
                ],
            )
            test_metrics = _evaluate_champion(
                test_context, champion, model_directory, resolved_model_id
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
                "rounds": rounds, "per_round": per_round, "pool_size": len(pool),
                "seeds": len(seeds), "validation_repeats": validation_repeats,
            },
            results_by_seed=results_by_seed,
            aggregate=aggregate,
            champion=champion,
            test_metrics=test_metrics,
            model_directory=model_directory,
            model_id=resolved_model_id,
        )
        summary = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "profile": profile.as_dict(),
            "budget": {
                "rounds": rounds, "per_round": per_round, "pool_size": len(pool),
                "seeds": list(seeds), "validation_repeats": validation_repeats,
            },
            "aggregate": aggregate,
            "champion": champion.candidate.describe() if champion is not None else None,
            "champion_seed": champion.seed if champion is not None else None,
            "champion_validation": (
                {
                    "macro_f1_mean": champion.val_macro_f1,
                    "macro_f1_std": champion.val_macro_f1_std,
                    "accuracy_mean": champion.val_accuracy,
                }
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
