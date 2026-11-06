"""Tool factory: let the LLM write a new feature extractor, then prove it works.

The generated code never touches the project: it is written to a scratch file,
checked with a small AST allowlist, executed in a **subprocess** with a timeout,
and only then evaluated against the project's own split protocol.  A candidate is
registered only if it beats the baseline by more than twice the split variation.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..io.dataset import load_dataset_records, load_record_signal, resolve_dataset_split
from ..training.pipeline import (
    PreparedDataset,
    evaluate_partition,
    fit_prepared,
    prepare_dataset,
)
from .tools import LabContext, RunExperimentArgs, run_experiment

ALLOWED_ROOTS = {"numpy", "scipy", "math"}
FORBIDDEN_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
FORBIDDEN_ATTRS = {"save", "savez", "tofile", "write", "dump", "load", "system", "popen"}
LABEL_ARGS = {"y", "label", "labels", "target", "targets"}


class Candidate(BaseModel):
    name: str = Field(min_length=3, max_length=40)
    rationale: str = Field(min_length=10)
    code: str = Field(min_length=30)


def validate_code(code: str) -> list[str]:
    """Static checks: imports, forbidden calls, and the extract() contract."""
    problems: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"语法错误：{error}"]

    has_extract = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_ROOTS:
                    problems.append(f"不允许导入 {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in ALLOWED_ROOTS:
                problems.append(f"不允许 from {node.module} import")
        elif isinstance(node, ast.FunctionDef) and node.name == "extract":
            has_extract = True
            if any(arg.arg in LABEL_ARGS for arg in node.args.args):
                problems.append("extract 不能接受标签参数")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"不允许使用 {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"不允许调用 .{node.attr}")
    if not has_extract:
        problems.append("缺少 extract(windows) 函数")
    return problems


RUNNER = '''
import importlib.util, json, sys
import numpy as np

code_path, win_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("candidate", code_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

windows = np.load(win_path)
first = np.asarray(module.extract(windows), dtype=float)
second = np.asarray(module.extract(windows), dtype=float)
report = {
    "shape": list(first.shape),
    "finite": bool(np.isfinite(first).all()),
    "deterministic": bool(np.array_equal(first, second)),
    "nontrivial": bool(np.nanstd(first, axis=0).max() > 1e-9) if first.size else False,
}
np.save(out_path, first)
print(json.dumps(report))
'''


@dataclass
class SandboxResult:
    ok: bool
    report: dict
    features: np.ndarray | None
    error: str = ""


def run_in_sandbox(code: str, windows: np.ndarray, timeout: float = 30.0) -> SandboxResult:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "candidate.py").write_text(code, encoding="utf-8")
        (base / "runner.py").write_text(RUNNER, encoding="utf-8")
        np.save(base / "windows.npy", windows)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(base / "runner.py"),
                    str(base / "candidate.py"),
                    str(base / "windows.npy"),
                    str(base / "features.npy"),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=base,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, {}, None, "执行超时")
        if completed.returncode != 0:
            return SandboxResult(
                False, {}, None, (completed.stderr or "执行失败")[-400:]
            )
        try:
            report = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return SandboxResult(False, {}, None, "无法解析运行结果")
        features = np.load(base / "features.npy")
        return SandboxResult(True, report, features)


def _windows_of(ctx: LabContext) -> np.ndarray:
    records = load_dataset_records(ctx.spec, ctx.data_root)
    width = ctx.spec.window.size
    windows = []
    for record in records:
        values = load_record_signal(record).values
        windows.append(values[:width])
    return np.stack(windows)


def evaluate_candidate(
    ctx: LabContext, features: np.ndarray, *, seeds: int = 3
) -> dict:
    """Train the project classifier on candidate features with the project split."""
    records = load_dataset_records(ctx.spec, ctx.data_root)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    matrix = features.reshape(len(records), 1, -1)
    per_seed = []
    for offset in range(seeds):
        seed = ctx.spec.split.seed + offset
        seeded = ctx.spec.model_copy(
            update={"split": ctx.spec.split.model_copy(update={"seed": seed})}
        )
        split_by_group = resolve_dataset_split(seeded, records)
        names = np.asarray([split_by_group[record.group] for record in records])
        masks = {name: names == name for name in ("train", "validation", "test")}
        prepared = PreparedDataset(
            spec=seeded,
            records=list(records),
            features=matrix,
            labels=labels,
            groups=tuple(record.group for record in records),
            feature_dimension=matrix.shape[2],
            split_by_group=split_by_group,
            masks=masks,
            feature_seconds=0.0,
        )
        model, _ = fit_prepared(prepared, seeded)
        metrics = evaluate_partition(
            prepared, model, "validation", healthy_class_id=seeded.labels.healthy_class_id
        )[0]
        per_seed.append(float(metrics["multiclass_macro_f1"]))
    values = np.asarray(per_seed)
    return {
        "seeds": seeds,
        "per_seed": [round(value, 6) for value in per_seed],
        "macro_f1_mean": round(float(values.mean()), 6),
        "macro_f1_std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 6),
    }


PROPOSAL_PROMPT = """You are designing a new feature extractor for vibration diagnosis.
Write ONE self-contained Python function with this exact signature:

    def extract(windows: np.ndarray) -> np.ndarray:
        # windows: shape (n, samples), float64, n recordings
        # return: shape (n, d), d >= 1, finite values, deterministic

Rules:
- only numpy / scipy / math; no file access, no network, no labels;
- the output must vary across windows (no constant columns);
- keep it under 60 lines and vectorised.

Reply as JSON: {"name": "...", "rationale": "...", "code": "..."}.
"""


def propose(llm, goal: str, history: str = "") -> Candidate:
    raw = llm(
        [
            {"role": "system", "content": PROPOSAL_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\nKnown results:\n{history}"},
        ]
    )
    text = raw.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    return Candidate.model_validate(json.loads(text))


REPAIR_PROMPT = """Your feature extractor failed when executed. Fix it.
Return the SAME JSON shape: {"name": "...", "rationale": "...", "code": "..."}.
Keep the signature `def extract(windows)`; only numpy/scipy/math; no labels,
no file access. Make sure the code runs on windows of shape (n, 2048) with
12 kHz sampling when you use frequency-based filters.
"""


def repair(llm, candidate: Candidate, error: str) -> Candidate:
    raw = llm(
        [
            {"role": "system", "content": REPAIR_PROMPT},
            {
                "role": "user",
                "content": f"Code:\n{candidate.code}\n\nError:\n{error[:800]}",
            },
        ]
    )
    text = raw.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    fixed = Candidate.model_validate(json.loads(text))
    return fixed


def build(
    ctx: LabContext, candidate: Candidate, *, seeds: int = 3, llm=None, retries: int = 1
) -> dict:
    windows = _windows_of(ctx)
    attempts: list[dict] = []
    sandbox = None
    for attempt in range(retries + 1):
        problems = validate_code(candidate.code)
        if problems:
            return {
                "accepted": False,
                "stage": "static",
                "problems": problems,
                "attempts": attempts,
            }
        sandbox = run_in_sandbox(candidate.code, windows)
        attempts.append(
            {
                "attempt": attempt,
                "ok": sandbox.ok,
                "error": None if sandbox.ok else sandbox.error[-300:],
            }
        )
        if sandbox.ok:
            break
        if llm is not None and attempt < retries:
            candidate = repair(llm, candidate, sandbox.error)
    if sandbox is None or not sandbox.ok:
        return {
            "accepted": False,
            "stage": "sandbox",
            "problems": [sandbox.error if sandbox else "no attempt"],
            "attempts": attempts,
        }
    report = sandbox.report
    if report["shape"][0] != len(windows) or len(report["shape"]) != 2:
        return {
            "accepted": False,
            "stage": "contract",
            "problems": [f"输出形状错误：{report['shape']}"],
        }
    for key, label in (
        ("finite", "输出包含 NaN/Inf"),
        ("deterministic", "同样输入两次结果不同"),
        ("nontrivial", "输出是常量，没有信息量"),
    ):
        if not report.get(key):
            return {"accepted": False, "stage": "contract", "problems": [label]}

    standalone = evaluate_candidate(ctx, sandbox.features, seeds=seeds)
    baseline = run_experiment(
        ctx,
        RunExperimentArgs(
            feature=ctx.spec.feature.method,
            classifier=ctx.spec.classifier.algorithm,
            seeds=seeds,
        ),
    )
    # A new tool has to *add* information, so it is evaluated on top of the
    # project's own feature set rather than as a replacement for it.
    prepared = prepare_dataset(ctx.spec, ctx.data_root)
    base_features = prepared.features.reshape(len(prepared.records), -1)
    combined = np.concatenate([base_features, sandbox.features], axis=1)
    metrics = evaluate_candidate(ctx, combined, seeds=seeds)
    # Paired comparison: both arms use the same split seeds, so the difficulty of
    # each split cancels out and the comparison is far more sensitive than the
    # difference of two independent means.
    baseline_per_seed = np.asarray(
        [row["macro_f1"] for row in baseline["per_seed"]], dtype=float
    )
    candidate_per_seed = np.asarray(metrics["per_seed"], dtype=float)
    count = min(len(baseline_per_seed), len(candidate_per_seed))
    per_seed_delta = candidate_per_seed[:count] - baseline_per_seed[:count]
    delta = float(per_seed_delta.mean())
    if count > 1:
        standard_error = float(per_seed_delta.std(ddof=1) / np.sqrt(count))
    else:
        standard_error = float("inf")
    threshold = 2 * standard_error
    accepted = bool(np.isfinite(threshold) and delta > threshold)
    result = {
        "accepted": accepted,
        "stage": "evaluated",
        "name": candidate.name,
        "rationale": candidate.rationale,
        "metrics": metrics,
        "standalone": standalone,
        "baseline": {
            "feature": baseline["feature"],
            "classifier": baseline["classifier"],
            "macro_f1_mean": baseline["macro_f1_mean"],
            "macro_f1_std": baseline["macro_f1_std"],
        },
        "delta": round(float(delta), 6),
        "threshold": round(float(threshold), 6),
        "per_seed_delta": [round(float(value), 6) for value in per_seed_delta],
        "attempts": attempts,
        "problems": []
        if accepted
        else ["配对差值没有显著大于 0（没有超过 2 倍标准误）"],
    }
    if accepted:
        tools_dir = ctx.workspace / "lab_tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        code_path = tools_dir / f"{candidate.name}.py"
        code_path.write_text(candidate.code, encoding="utf-8")
        registry = tools_dir / "registry.json"
        entries = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
        entries.append(
            {
                "name": candidate.name,
                "code": code_path.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "macro_f1_mean": metrics["macro_f1_mean"],
                "macro_f1_std": metrics["macro_f1_std"],
                "baseline_mean": baseline["macro_f1_mean"],
                "delta": result["delta"],
                "per_seed_delta": result["per_seed_delta"],
            }
        )
        registry.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["registered"] = str(code_path)
    return result
