"""Ground hypotheses in tool evidence.

An LLM can write a fluent hypothesis that cites numbers it never saw.  These
checks reject that: a hypothesis must cite at least one measured number that
actually appears in the experiment history, and must state a testable prediction.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
PREDICTION_WORDS = (
    "预计",
    "预期",
    "预测",
    "应该",
    "应当",
    "说明",
    "will",
    "expect",
    "should",
    "predict",
    "hypothes",
)


def numbers_in(text: str) -> list[float]:
    return [float(match.group()) for match in NUMBER.finditer(text)]


def _collect(value, out: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.startswith("_"):
                continue
            _collect(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, out)


def history_numbers(history: list[dict]) -> list[float]:
    out: list[float] = []
    for item in history:
        _collect({key: value for key, value in item.items() if key != "_signature"}, out)
    return out


def _is_grounded(number: float, measured: list[float]) -> bool:
    for value in measured:
        if abs(value - number) <= max(1e-9, abs(value) * 0.02):
            return True
        if abs(value) >= 10 and str(value).startswith(str(number)):
            return True
    return False


def _needs_grounding(number: float) -> bool:
    # Integers below 100 are usually counts of classes/seeds, not measurements.
    return not float(number).is_integer() or abs(number) >= 100


def problems(text: str, history: list[dict]) -> list[str]:
    if not text.strip():
        return []
    found: list[str] = []
    numbers = numbers_in(text)
    measured = history_numbers(history)
    unsupported = (
        []
        if not measured
        else [
            number
            for number in numbers
            if _needs_grounding(number) and not _is_grounded(number, measured)
        ]
    )
    if unsupported:
        found.append(f"引用了实验结果里没有的数字：{unsupported[:5]}")
    if not any(word in text.lower() for word in PREDICTION_WORDS):
        found.append("缺少可检验的预测（需要‘预计/预期/will’之类的表述）")
    if len(text) > 300:
        found.append("假设过长，压缩到 2 句以内")
    return found
