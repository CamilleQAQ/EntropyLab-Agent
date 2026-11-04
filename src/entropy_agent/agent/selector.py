"""Deterministic candidate ranking for the 5-round Agent.

There is no language model and no randomness here: the same history and pool
always produce the same next batch.  The policy is intentionally simple:

* round 0 explores families and classifiers, starting from the anchor combo;
* round 1 keeps covering unused families but orders them by validation score;
* later rounds exploit the best method/classifier seen so far and reserve one
  slot for a family that has not been tried yet.

Selection reads validation metrics only.  Test metrics are never passed in.
"""

from __future__ import annotations

from collections.abc import Sequence

from .candidates import Candidate

ANCHOR_METHOD = "TSMEFuDE"
ANCHOR_CLASSIFIER = "BT-LSTSVM"


def _aggregate(history: Sequence[dict], key: str) -> dict[str, float]:
    """Best validation macro-F1 per method or classifier, failures ignored."""
    best: dict[str, float] = {}
    for entry in history:
        if entry.get("status") != "OK" or entry.get("val_macro_f1") is None:
            continue
        name = entry.get(key)
        best[name] = max(best.get(name, float("-inf")), float(entry["val_macro_f1"]))
    return best


def _explore_order(
    remaining: Sequence[Candidate],
    history: Sequence[dict],
    anchor_id: str | None,
    limit: int,
) -> list[Candidate]:
    """Greedy coverage: anchor first, then an unseen family with an unseen classifier.

    The used-family/used-classifier sets grow as the batch is built, so one batch
    never wastes two slots on the same family.
    """
    used_families = {entry["family"] for entry in history}
    used_classifiers = {entry["classifier"] for entry in history}
    candidates = list(remaining)
    ordered: list[Candidate] = []
    while candidates and len(ordered) < limit:
        best = min(
            candidates,
            key=lambda candidate: (
                0 if candidate.candidate_id == anchor_id and not ordered else 1,
                0 if candidate.family not in used_families else 1,
                0 if candidate.classifier.algorithm not in used_classifiers else 1,
                candidates.index(candidate),
            ),
        )
        ordered.append(best)
        candidates.remove(best)
        used_families.add(best.family)
        used_classifiers.add(best.classifier.algorithm)
    return ordered


def _coverage_then_score(
    remaining: Sequence[Candidate], history: Sequence[dict], limit: int
) -> list[Candidate]:
    used_families = {entry["family"] for entry in history}
    method_score = _aggregate(history, "method")
    classifier_score = _aggregate(history, "classifier")
    candidates = list(remaining)
    ordered: list[Candidate] = []
    while candidates and len(ordered) < limit:
        best = min(
            candidates,
            key=lambda candidate: (
                0 if candidate.family not in used_families else 1,
                -method_score.get(candidate.method.method, -1.0),
                -classifier_score.get(candidate.classifier.algorithm, -1.0),
                candidates.index(candidate),
            ),
        )
        ordered.append(best)
        candidates.remove(best)
        used_families.add(best.family)
    return ordered


def _exploit_order(
    remaining: Sequence[Candidate],
    history: Sequence[dict],
    limit: int,
) -> list[Candidate]:
    used_families = {entry["family"] for entry in history}
    method_score = _aggregate(history, "method")
    classifier_score = _aggregate(history, "classifier")

    def score(candidate: Candidate) -> float:
        return (
            2.0 * method_score.get(candidate.method.method, -1.0)
            + classifier_score.get(candidate.classifier.algorithm, -1.0)
        )

    ordered = [
        candidate
        for _, candidate in sorted(
            enumerate(remaining), key=lambda item: (-score(item[1]), item[0])
        )
    ]
    if limit <= 2:
        return ordered[:limit]

    exploitation = ordered[: limit - 1]
    chosen = {candidate.candidate_id for candidate in exploitation}
    exploration = next(
        (
            candidate
            for candidate in ordered
            if candidate.candidate_id not in chosen and candidate.family not in used_families
        ),
        None,
    )
    if exploration is None:
        exploration = next(
            candidate for candidate in ordered if candidate.candidate_id not in chosen
        )
    return (exploitation + [exploration])[:limit]


def propose(
    round_index: int,
    history: Sequence[dict],
    pool: Sequence[Candidate],
    per_round: int = 3,
    anchor_id: str | None = None,
) -> tuple[Candidate, ...]:
    """Return the next batch of untried candidates for this round."""
    if per_round < 1:
        raise ValueError("per_round must be positive")
    tried = {entry["candidate_id"] for entry in history}
    remaining = [candidate for candidate in pool if candidate.candidate_id not in tried]
    if not remaining:
        return ()
    if round_index == 0:
        ordered = _explore_order(remaining, history, anchor_id, per_round)
    elif round_index == 1:
        ordered = _coverage_then_score(remaining, history, per_round)
    else:
        ordered = _exploit_order(remaining, history, per_round)
    return tuple(ordered[:per_round])


def anchor_candidate(pool: Sequence[Candidate]) -> Candidate | None:
    for candidate in pool:
        if (
            candidate.method.method == ANCHOR_METHOD
            and candidate.classifier.algorithm == ANCHOR_CLASSIFIER
        ):
            return candidate
    return None
