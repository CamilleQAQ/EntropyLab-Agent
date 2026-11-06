"""Tools the research lab can call.

Every tool is a plain function over a shared :class:`LabContext`; the planner sees
only the name, description and JSON schema.  Tools reuse the project's existing
data/feature/classifier code so lab results are the same numbers as the CLI.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..agent.candidates import DEFAULT_CLASSIFIERS, DEFAULT_METHODS
from ..io.dataset import DatasetSpec, load_dataset_records
from ..training.pipeline import evaluate_partition, prepare_dataset, train_dataset


@dataclass
class LabContext:
    spec: DatasetSpec
    data_root: Path
    workspace: Path
    goal: str = ""
    budget_seconds: float = 600.0
    spent_seconds: float = 0.0
    experiments: list[dict] = field(default_factory=list)

    def model_dir(self, tag: str) -> Path:
        path = self.workspace / "models" / f"{tag}-{uuid.uuid4().hex[:8]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class ProfileArgs(BaseModel):
    pass


class RunExperimentArgs(BaseModel):
    feature: str = Field(description="feature method, e.g. FUSION_TIME, TSMEFuDE")
    classifier: str = Field(description="classifier, e.g. EXTRA-TREES, SVM-RBF")
    seeds: int = Field(default=5, ge=1, le=10, description="number of split seeds")


class AnalyzeErrorsArgs(BaseModel):
    feature: str
    classifier: str
    seed: int = Field(default=20261104)


class SearchNotesArgs(BaseModel):
    query: str = Field(description="what to look up, e.g. inner ring misclassification")


def _method_spec(name: str):
    for method in DEFAULT_METHODS:
        if method.method == name:
            return method
    raise ValueError(f"unknown feature method {name!r}")


def _classifier_spec(name: str):
    for classifier in DEFAULT_CLASSIFIERS:
        if classifier.algorithm == name:
            return classifier
    raise ValueError(f"unknown classifier {name!r}")


def _variant(ctx: LabContext, feature: str, classifier: str) -> DatasetSpec:
    method = _method_spec(feature)
    clf = _classifier_spec(classifier)
    return ctx.spec.model_copy(
        update={
            "feature": ctx.spec.feature.model_copy(
                update={"method": method.method, "parameters": dict(method.parameters)}
            ),
            "classifier": ctx.spec.classifier.model_copy(
                update={
                    "algorithm": clf.algorithm,
                    "backend": clf.backend,
                    "parameters": dict(clf.parameters),
                }
            ),
        }
    )


def profile_dataset(ctx: LabContext, _args: ProfileArgs | None = None) -> dict:
    started = time.perf_counter()
    records = load_dataset_records(ctx.spec, ctx.data_root)
    labels = np.asarray([record.label for record in records])
    groups = {record.group for record in records}
    per_class = {name: int((labels == index).sum()) for index, name in enumerate(ctx.spec.labels.names)}
    result = {
        "dataset_id": ctx.spec.dataset_id,
        "records": len(records),
        "classes": list(ctx.spec.labels.names),
        "records_per_class": per_class,
        "groups": len(groups),
        "sampling_rate_hz": ctx.spec.sampling_rate_hz,
        "window": ctx.spec.window.size,
        "windows_per_record": ctx.spec.window.windows_per_record,
        "group_size": ctx.spec.layout.group_size,
        "balanced": len(set(per_class.values())) == 1,
    }
    ctx.spent_seconds += time.perf_counter() - started
    ctx.experiments.append({"tool": "profile_dataset", "result": result})
    return result


def run_experiment(ctx: LabContext, args: RunExperimentArgs) -> dict:
    started = time.perf_counter()
    spec = _variant(ctx, args.feature, args.classifier)
    base_seed = spec.split.seed
    per_seed = []
    for offset in range(args.seeds):
        seed = base_seed + offset
        seeded = spec.model_copy(
            update={"split": spec.split.model_copy(update={"seed": seed})}
        )
        prepared = prepare_dataset(seeded, ctx.data_root)
        result = train_dataset(
            seeded,
            ctx.data_root,
            ctx.model_dir("lab"),
            model_id=f"lab-{seeded.dataset_id.lower()}-{seed}",
            prepared=prepared,
            evaluate_test=False,
        )
        validation = result.test_result["validation"]
        per_seed.append(
            {
                "seed": seed,
                "accuracy": round(float(validation["multiclass_accuracy"]), 6),
                "macro_f1": round(float(validation["multiclass_macro_f1"]), 6),
            }
        )
    f1s = np.asarray([row["macro_f1"] for row in per_seed])
    accs = np.asarray([row["accuracy"] for row in per_seed])
    summary = {
        "tool": "run_experiment",
        "feature": args.feature,
        "classifier": args.classifier,
        "seeds": args.seeds,
        "macro_f1_mean": round(float(f1s.mean()), 6),
        "macro_f1_std": round(float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0, 6),
        "accuracy_mean": round(float(accs.mean()), 6),
        "per_seed": per_seed,
        "seconds": round(time.perf_counter() - started, 2),
    }
    ctx.spent_seconds += summary["seconds"]
    ctx.experiments.append(summary)
    return summary


def analyze_errors(ctx: LabContext, args: AnalyzeErrorsArgs) -> dict:
    from sklearn.metrics import confusion_matrix

    started = time.perf_counter()
    spec = _variant(ctx, args.feature, args.classifier)
    seeded = spec.model_copy(
        update={"split": spec.split.model_copy(update={"seed": args.seed})}
    )
    prepared = prepare_dataset(seeded, ctx.data_root)
    result = train_dataset(
        seeded,
        ctx.data_root,
        ctx.model_dir("err"),
        model_id=f"lab-errors-{args.seed}",
        prepared=prepared,
        evaluate_test=False,
    )
    metrics, predictions, _, _ = evaluate_partition(
        prepared,
        result.artifact.model,
        "validation",
        healthy_class_id=seeded.labels.healthy_class_id,
    )
    truth = prepared.labels[prepared.masks["validation"]]
    names = list(seeded.labels.names)
    matrix = confusion_matrix(truth, predictions, labels=range(len(names))).tolist()
    per_class = {
        name: (round(matrix[i][i] / sum(matrix[i]), 4) if sum(matrix[i]) else None)
        for i, name in enumerate(names)
    }
    pairs = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j and matrix[i][j] > 0:
                pairs.append({"truth": name_i, "predicted": name_j, "count": matrix[i][j]})
    pairs.sort(key=lambda item: item["count"], reverse=True)
    summary = {
        "tool": "analyze_errors",
        "feature": args.feature,
        "classifier": args.classifier,
        "seed": args.seed,
        "confusion": matrix,
        "labels": names,
        "per_class_recall": per_class,
        "top_confusions": pairs[:5],
        "macro_f1": round(float(metrics["multiclass_macro_f1"]), 6),
        "seconds": round(time.perf_counter() - started, 2),
    }
    ctx.spent_seconds += summary["seconds"]
    ctx.experiments.append(summary)
    return summary


NOTES_DIR = Path(__file__).resolve().parents[3] / "docs"


def search_notes(ctx: LabContext, args: SearchNotesArgs) -> dict:
    terms = [term for term in re.split(r"\W+", args.query.lower()) if len(term) > 2]
    scored = []
    if NOTES_DIR.is_dir():
        for path in sorted(NOTES_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            paragraphs = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
            for paragraph in paragraphs:
                haystack = paragraph.lower()
                score = sum(haystack.count(term) for term in terms)
                if score:
                    scored.append((score, path.name, paragraph[:400]))
    scored.sort(key=lambda item: item[0], reverse=True)
    result = {
        "tool": "search_notes",
        "query": args.query,
        "hits": [
            {"file": name, "score": score, "text": text}
            for score, name, text in scored[:3]
        ],
    }
    ctx.experiments.append(result)
    return result


TOOL_SPECS = {
    "profile_dataset": {
        "description": "Read the dataset and return classes, counts, groups, window and sampling rate.",
        "args": ProfileArgs,
        "fn": profile_dataset,
    },
    "run_experiment": {
        "description": "Train a (feature, classifier) pair over N split seeds and return validation macro-F1 mean/std.",
        "args": RunExperimentArgs,
        "fn": run_experiment,
    },
    "analyze_errors": {
        "description": "Confusion matrix, per-class recall and the most confused class pairs on validation.",
        "args": AnalyzeErrorsArgs,
        "fn": analyze_errors,
    },
    "search_notes": {
        "description": "Search the project's method/data notes for evidence before proposing a hypothesis.",
        "args": SearchNotesArgs,
        "fn": search_notes,
    },
}


def tool_catalogue() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        lines.append(f"- {name}: {spec['description']}")
        lines.append(f"  args schema: {json.dumps(spec['args'].model_json_schema(), ensure_ascii=False)}")
    return "\n".join(lines)
