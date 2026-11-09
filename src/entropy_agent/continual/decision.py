"""The promotion decision, separated from the training loop.

The probe rule splits the decision into a guardrail and an objective:

* **guardrail** - retention on probe sets that were never trained (the worst drop
  over the reference-domain validation set and every previous batch's probes);
* **objective**  - gain on the new batch's probe.

The fixed validation margin is available as a comparison baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    promoted: bool
    reason: str
    retention_drop: float | None = None
    gain_new: float | None = None
    # Three-state gate: publish / defer / reject.  `defer` keeps the
    # challenger alive for re-judgement when more labels arrive; `rejected`
    # drops it.
    status: str = "publish"


def decide(
    *,
    criterion: str,
    champion_validation_macro_f1: float,
    challenger_validation_macro_f1: float,
    margin: float,
    retention_drop: float | None = None,
    gain_new: float | None = None,
    guard_tolerance: float = 0.02,
    min_gain: float = 0.0,
) -> Decision:
    """Return the promotion decision and a human-readable reason."""
    if criterion == "margin":
        threshold = champion_validation_macro_f1 + margin
        promoted = challenger_validation_macro_f1 >= threshold
        return Decision(
            promoted=promoted,
            reason=(
                f"{challenger_validation_macro_f1:.4f} vs "
                f"{champion_validation_macro_f1:.4f} "
                f"(needs {threshold:.4f} = champion + margin {margin})"
            ),
        )
    if criterion != "guardrail":
        raise ValueError(f"unknown criterion {criterion!r}")

    if retention_drop is None or gain_new is None:
        raise ValueError("guardrail criterion needs retention_drop and gain_new")
    if retention_drop > 2 * guard_tolerance:
        return Decision(
            promoted=False,
            reason=(
                f"retention drop {retention_drop:.4f} > 2x tolerance {2 * guard_tolerance:.4f} "
                f"(new probe gain {gain_new:+.4f})"
            ),
            retention_drop=retention_drop,
            gain_new=gain_new,
            status="reject",
        )
    if retention_drop > guard_tolerance:
        if gain_new > min_gain:
            return Decision(
                promoted=False,
                reason=(
                    f"trade-off: new gain {gain_new:+.4f} vs retention drop "
                    f"{retention_drop:.4f}; deferring for re-judgement"
                ),
                retention_drop=retention_drop,
                gain_new=gain_new,
                status="defer",
            )
        return Decision(
            promoted=False,
            reason=(
                f"retention drop {retention_drop:.4f} > tolerance {guard_tolerance:.4f} "
                f"without new-batch gain {gain_new:+.4f}"
            ),
            retention_drop=retention_drop,
            gain_new=gain_new,
            status="reject",
        )
    if gain_new <= min_gain:
        return Decision(
            promoted=False,
            status="reject",
            reason=(
                f"new-batch gain {gain_new:+.4f} <= minimum {min_gain:+.4f} "
                f"(retention drop {retention_drop:.4f})"
            ),
            retention_drop=retention_drop,
            gain_new=gain_new,
        )
    return Decision(
        promoted=True,
        reason=(
            f"new-batch gain {gain_new:+.4f} > {min_gain:+.4f} and "
            f"retention drop {retention_drop:.4f} <= {guard_tolerance:.4f}"
        ),
        retention_drop=retention_drop,
        gain_new=gain_new,
    )
