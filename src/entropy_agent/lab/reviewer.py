"""Deterministic reviewer: blocks claims the experiment cannot support.

The LLM proposes and interprets; this module decides whether the evidence is
strong enough.  Keeping it rule-based means the gate is testable and cannot be
talked around by a persuasive model.
"""

from __future__ import annotations

import math

import numpy as np

MIN_SEEDS = 3
MAX_STD = 0.05


def _paired_delta(result: dict, baseline: dict) -> tuple[float, float] | None:
    """Mean per-split difference and its standard error, when both arms have it."""
    def values(entry: dict) -> list[float]:
        rows = entry.get("per_seed") or []
        return [
            float(row["macro_f1"]) if isinstance(row, dict) else float(row)
            for row in rows
        ]

    candidate, reference = values(result), values(baseline)
    count = min(len(candidate), len(reference))
    if count < 2:
        return None
    deltas = np.asarray(candidate[:count]) - np.asarray(reference[:count])
    standard_error = float(deltas.std(ddof=1) / math.sqrt(count))
    return float(deltas.mean()), standard_error


def review_experiment(result: dict, baseline: dict | None, claim: str) -> dict:
    findings: list[dict] = []
    if result.get("error"):
        return {
            "verdict": "revise",
            "findings": [{"code": "TOOL_FAILED", "detail": str(result["error"])[:160]}],
        }
    if result.get("tool") != "run_experiment":
        return {"verdict": "accept", "findings": findings}

    seeds = int(result.get("seeds") or 1)
    std = float(result.get("macro_f1_std") or 0.0)
    mean = float(result.get("macro_f1_mean") or 0.0)

    if seeds < MIN_SEEDS:
        findings.append(
            {
                "code": "TOO_FEW_SEEDS",
                "detail": f"only {seeds} split seed(s); at least {MIN_SEEDS} are required",
            }
        )
    if std > MAX_STD:
        findings.append(
            {
                "code": "UNSTABLE",
                "detail": f"macro-F1 std {std:.4f} exceeds {MAX_STD:.2f} across splits",
            }
        )
    if claim and baseline is not None:
        paired = _paired_delta(result, baseline)
        if paired is None:
            delta = mean - float(baseline.get("macro_f1_mean") or 0.0)
            pooled = max(std, float(baseline.get("macro_f1_std") or 0.0))
            significant = abs(delta) >= 2 * pooled
            detail_basis = f"2x split variation ({2 * pooled:.4f})"
            delta_value = delta
        else:
            delta_value, standard_error = paired
            significant = delta_value >= 2 * standard_error
            detail_basis = f"2x paired standard error ({2 * standard_error:.4f})"
        if not significant:
            findings.append(
                {
                    "code": "CLAIM_WITHIN_NOISE",
                    "detail": (
                        f"claimed change {delta_value:+.4f} is smaller than {detail_basis}"
                    ),
                }
            )
    verdict = "revise" if findings else "accept"
    return {"verdict": verdict, "findings": findings}


def review_plan(action: str, rounds: int, max_rounds: int, spent: float, budget: float) -> dict:
    findings: list[dict] = []
    if rounds >= max_rounds:
        findings.append({"code": "ROUND_BUDGET", "detail": f"reached {max_rounds} rounds"})
    if spent >= budget:
        findings.append(
            {"code": "TIME_BUDGET", "detail": f"spent {spent:.0f}s of {budget:.0f}s"}
        )
    return {"verdict": "stop" if findings else "continue", "findings": findings}
