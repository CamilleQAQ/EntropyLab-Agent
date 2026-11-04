"""Deterministic, cost-aware candidate ranking for the 5-round Agent.

There is no language model and no randomness: the same history and pool always
produce the same batch, and every proposal carries a human-readable reason.

Policy:

* round 0 explores the anchor combination first, then unused families paired with
  unused classifiers;
* round 1 keeps covering unused families, ordered by the best validation score of
  each method seen so far;
* later rounds exploit: methods whose features are already in the cache come
  first (their classifiers are free), then an untried classifier for that method
  (the cheap unknown) before re-running a classifier that already has a score,
  then the best validation score, and one slot is still reserved for an
  untouched family.

Selection reads validation metrics only.  Test metrics are never passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .candidates import Candidate

ANCHOR_METHOD = "TSMEFuDE"
ANCHOR_CLASSIFIER = "BT-LSTSVM"


@dataclass(frozen=True)
class Proposal:
    candidate: Candidate
    reason: str


def _aggregate(history: Sequence[dict], key: str) -> dict[str, float]:
    """Best validation macro-F1 per method or classifier, failures ignored."""
    best: dict[str, float] = {}
    for entry in history:
        value = entry.get("val_macro_f1")
        if entry.get("status") != "OK" or value is None:
            continue
        name = entry.get(key)
        best[name] = max(best.get(name, float("-inf")), float(value))
    return best


def _explore(
    remaining: Sequence[Candidate],
    history: Sequence[dict],
    anchor_id: str | None,
    limit: int,
) -> list[Proposal]:
    used_families = {entry["family"] for entry in history}
    used_classifiers = {entry["classifier"] for entry in history}
    candidates = list(remaining)
    proposals: list[Proposal] = []
    while candidates and len(proposals) < limit:
        best = min(
            candidates,
            key=lambda candidate: (
                0 if candidate.candidate_id == anchor_id and not proposals else 1,
                0 if candidate.family not in used_families else 1,
                0 if candidate.classifier.algorithm not in used_classifiers else 1,
                candidates.index(candidate),
            ),
        )
        if best.candidate_id == anchor_id and not proposals:
            reason = "anchor combination"
        elif best.family not in used_families:
            reason = f"unvisited family {best.family}"
        else:
            reason = "unvisited classifier in a visited family"
        proposals.append(Proposal(best, reason))
        candidates.remove(best)
        used_families.add(best.family)
        used_classifiers.add(best.classifier.algorithm)
    return proposals


def _cover_then_score(
    remaining: Sequence[Candidate], history: Sequence[dict], limit: int
) -> list[Proposal]:
    used_families = {entry["family"] for entry in history}
    method_score = _aggregate(history, "method")
    candidates = list(remaining)
    proposals: list[Proposal] = []
    while candidates and len(proposals) < limit:
        best = min(
            candidates,
            key=lambda candidate: (
                0 if candidate.family not in used_families else 1,
                -method_score.get(candidate.method.method, -1.0),
                candidates.index(candidate),
            ),
        )
        score = method_score.get(best.method.method)
        if best.family not in used_families:
            reason = f"cover family {best.family}"
        elif score is not None:
            reason = f"best method so far ({best.method.method}, val {score:.4f})"
        else:
            reason = "first classifier for this method"
        proposals.append(Proposal(best, reason))
        candidates.remove(best)
        used_families.add(best.family)
    return proposals


def _exploit(
    remaining: Sequence[Candidate],
    history: Sequence[dict],
    extracted_methods: set[str],
    limit: int,
) -> list[Proposal]:
    method_score = _aggregate(history, "method")
    used_families = {entry["family"] for entry in history}

    def sort_key(item: tuple[int, Candidate]) -> tuple:
        index, candidate = item
        free = 0 if candidate.method.method in extracted_methods else 1
        # Cost-optimal order: a method whose features are already extracted has
        # free classifiers, so it is swept first; among those, the method with the
        # best validation score so far goes first, and its remaining classifiers
        # are taken in pool order (no classifier score is reused here: repeating a
        # classifier that scored well with another method is not free insight).
        return (free, -method_score.get(candidate.method.method, -1.0), index)

    ordered = [candidate for _, candidate in sorted(enumerate(remaining), key=sort_key)]
    if limit <= 2:
        chosen = ordered[:limit]
    else:
        exploitation = ordered[: limit - 1]
        chosen_ids = {candidate.candidate_id for candidate in exploitation}
        exploration = next(
            (
                candidate
                for candidate in ordered
                if candidate.candidate_id not in chosen_ids
                and candidate.family not in used_families
            ),
            None,
        )
        if exploration is None:
            exploration = next(
                candidate
                for candidate in ordered
                if candidate.candidate_id not in chosen_ids
            )
        chosen = exploitation + ([exploration] if exploration else [])

    proposals: list[Proposal] = []
    for candidate in chosen:
        score = method_score.get(candidate.method.method)
        if candidate.method.method in extracted_methods:
            verdict = "free features" if score is None else f"free features, val {score:.4f}"
            reason = f"sweep {candidate.classifier.algorithm} for {candidate.method.method} ({verdict})"
        else:
            reason = f"new extraction: {candidate.method.method} ({candidate.family})"
        proposals.append(Proposal(candidate, reason))
    return proposals


def propose(
    round_index: int,
    history: Sequence[dict],
    pool: Sequence[Candidate],
    per_round: int = 3,
    anchor_id: str | None = None,
    extracted_methods: set[str] | None = None,
) -> tuple[Proposal, ...]:
    """Return the next batch of untried candidates, each with its reason."""
    if per_round < 1:
        raise ValueError("per_round must be positive")
    extracted = set(extracted_methods or ())
    tried = {entry["candidate_id"] for entry in history}
    remaining = [candidate for candidate in pool if candidate.candidate_id not in tried]
    if not remaining:
        return ()
    if round_index == 0:
        return tuple(_explore(remaining, history, anchor_id, per_round))
    if round_index == 1:
        return tuple(_cover_then_score(remaining, history, per_round))
    return tuple(_exploit(remaining, history, extracted, per_round))


def anchor_candidate(pool: Sequence[Candidate]) -> Candidate | None:
    for candidate in pool:
        if (
            candidate.method.method == ANCHOR_METHOD
            and candidate.classifier.algorithm == ANCHOR_CLASSIFIER
        ):
            return candidate
    return None
