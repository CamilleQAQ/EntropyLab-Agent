"""When to stop paying for more candidates (the `search` action).

Without a stopping policy the round simply spends its whole budget: that is a
fixed rule, not a decision.  With one, every paid evaluation is followed by an
explicit "continue or commit" choice, which is what turns the budget from a
hyper-parameter into an action.

Two interpretable rules, each of which can be disabled:

* **lead** - stop once the best feasible candidate beats the runner-up by at least
  ``min_lead``; further search cannot plausibly change the winner;
* **patience** - stop after ``patience`` consecutive candidates that fail to
  improve the best result.

Both are deliberately simple: with a handful of rounds there is not enough data to
learn a stopping policy, and an explainable rule can be ablated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchState:
    evaluated: int
    best_gain: float | None
    runner_up_gain: float | None
    no_improve: int
    remaining_seconds: float


@dataclass(frozen=True)
class SearchDecision:
    stop: bool
    reason: str


def should_stop(state: SearchState, policy: "SearchPolicy") -> SearchDecision:
    if not policy.enabled:
        return SearchDecision(False, "")
    if state.evaluated < policy.min_candidates:
        return SearchDecision(False, "")
    if state.best_gain is None:
        if policy.patience and state.no_improve >= policy.patience:
            return SearchDecision(
                True, f"no feasible candidate after {state.no_improve} evaluations"
            )
        return SearchDecision(False, "")
    if state.runner_up_gain is None:
        return SearchDecision(False, "")
    lead = state.best_gain - state.runner_up_gain
    if policy.min_lead > 0 and lead >= policy.min_lead:
        return SearchDecision(
            True,
            f"clear lead: best gain {state.best_gain:+.4f} vs runner-up "
            f"{state.runner_up_gain:+.4f}",
        )
    if policy.patience and state.no_improve >= policy.patience:
        return SearchDecision(
            True,
            f"no improvement in the last {state.no_improve} paid candidates",
        )
    return SearchDecision(False, "")


@dataclass(frozen=True)
class SearchPolicy:
    enabled: bool = False
    min_lead: float = 0.02
    patience: int = 6
    min_candidates: int = 3

    @classmethod
    def adaptive(
        cls, *, min_lead: float = 0.02, patience: int = 6, min_candidates: int = 3
    ) -> "SearchPolicy":
        return cls(
            enabled=True,
            min_lead=min_lead,
            patience=patience,
            min_candidates=min_candidates,
        )

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "min_lead": self.min_lead,
            "patience": self.patience,
            "min_candidates": self.min_candidates,
        }
