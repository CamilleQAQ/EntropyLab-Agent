"""Cost model for feature extractions, and the budget accounting built on it.

A unit of the Agent is one entropy method (or one fusion recipe) with all of its
classifiers.  Those units differ in price by two orders of magnitude — a fusion
recipe is ~5 ms per window while TSMEFuDE is ~500 ms — so counting units made the
budget meaningless.  This module prices a unit in seconds instead.

Price = per-window prior (measured on this repository's 1892-2000 point windows)
times the number of windows the dataset will featurize.  Observations update the
prior with exponential smoothing, and `from_memory` seeds the model from earlier
runs, which is why a new dataset starts from a better guess than a hardcoded
table.

Budget is charged with the *estimate*, not with wall-clock time, so a warm
feature cache cannot make two arms incomparable: an arm pays for the extraction it
would have needed on a cold cache.  Actual extraction seconds are still recorded
per trial.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# Seconds per window, measured on 1892-2000 point windows in this repository.
PRIOR_SECONDS_PER_WINDOW: dict[str, float] = {
    "TSMEFuDE": 0.52,
    "MFE": 0.41,
    "RCMFuDE": 0.14,
    "MFuDE": 0.029,
    "IMFRDE": 0.018,
    "RCMDE": 0.014,
    "MDE": 0.0015,
    "MPE": 0.0011,
    "FUSION_TIME": 0.003,
    "FUSION_TIME_SPEC": 0.005,
    "FUSION_TIME_ENVELOPE": 0.005,
    "FUSION_TIME_ENVELOPE_ENTROPY": 0.005,
    "FUSION_ALL": 0.008,
}

DEFAULT_SECONDS_PER_WINDOW = 0.05


@dataclass
class CostModel:
    """Per-window price estimates, updated as the run pays for extractions."""

    per_window: dict[str, float] = field(
        default_factory=lambda: dict(PRIOR_SECONDS_PER_WINDOW)
    )
    observations: dict[str, int] = field(default_factory=dict)

    def estimate(self, method: str, windows: int) -> float:
        price = self.per_window.get(method, DEFAULT_SECONDS_PER_WINDOW)
        return max(price, 1e-6) * max(int(windows), 1)

    def observe(self, method: str, windows: int, seconds: float, weight: float = 0.5) -> None:
        """Blend a real measurement into the estimate; ignore cached zero costs."""
        if seconds <= 0 or windows <= 0:
            return
        observed = seconds / windows
        previous = self.per_window.get(method, DEFAULT_SECONDS_PER_WINDOW)
        self.per_window[method] = (1 - weight) * previous + weight * observed
        self.observations[method] = self.observations.get(method, 0) + 1

    @classmethod
    def from_memory(cls, path: Path | str | None) -> "CostModel":
        """Seed the estimates from earlier runs recorded in an experiment memory file."""
        model = cls()
        if path is None:
            return model
        path = Path(path)
        if not path.is_file():
            return model
        try:
            connection = sqlite3.connect(path)
            rows = connection.execute(
                "SELECT method, extract_seconds, extracted_records FROM trials "
                "WHERE cache_hit = 0 AND extract_seconds > 0 AND extracted_records > 0"
            ).fetchall()
            connection.close()
        except sqlite3.Error:
            return model
        samples: dict[str, list[float]] = {}
        for method, seconds, records in rows:
            samples.setdefault(method, []).append(float(seconds) / float(records))
        for method, values in samples.items():
            model.per_window[method] = float(median(values))
        return model

    def as_dict(self) -> dict:
        return {
            "per_window_seconds": {key: round(value, 6) for key, value in sorted(self.per_window.items())},
            "estimates": {"observations": self.observations},
        }


@dataclass
class BudgetLedger:
    """How one arm spent its seconds."""

    budget_seconds: float
    spent_seconds: float = 0.0
    paid_units: list[str] = field(default_factory=list)
    skipped_units: list[dict] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return self.budget_seconds - self.spent_seconds

    def can_afford(self, seconds: float) -> bool:
        return seconds <= self.remaining + 1e-9

    def charge(self, method: str, seconds: float) -> None:
        self.spent_seconds += float(seconds)
        self.paid_units.append(method)

    def skip(self, method: str, seconds: float, reason: str) -> None:
        self.skipped_units.append(
            {"method": method, "estimated_seconds": round(seconds, 3), "reason": reason}
        )

    def as_dict(self) -> dict:
        return {
            "budget_seconds": round(self.budget_seconds, 3),
            "spent_seconds": round(self.spent_seconds, 3),
            "remaining_seconds": round(self.remaining, 3),
            "paid_units": list(self.paid_units),
            "skipped_units": list(self.skipped_units),
        }


def save_cost_model(path: Path, model: CostModel, ledgers: dict[str, BudgetLedger]) -> None:
    payload = {
        "cost_model": model.as_dict(),
        "budgets": {arm: ledger.as_dict() for arm, ledger in ledgers.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
