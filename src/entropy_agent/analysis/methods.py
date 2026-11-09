"""Catalogue of entropy methods available to the planner.

This is a policy subset, not a second implementation of the numerical toolbox.
The bundled toolbox owns the eight entropy methods and every calculation.

Two catalogues live here and they are deliberately kept apart:

* ``METHOD_FAMILIES`` / ``V1_AUTO_METHODS`` mirror the entropy toolbox registry
  exactly (a test asserts that equality), so they must not grow with feature
  families that the toolbox does not implement;
* ``FUSION_FAMILIES`` names the deterministic time/frequency/envelope recipes
  from :mod:`entropy_agent.analysis.feature_pipeline`.

``FEATURE_FAMILIES`` is the union the Agent reasons about.
"""

from __future__ import annotations

from typing import Literal

AgentMethod = Literal[
    "MDE", "MPE", "MFE", "MFuDE", "TSMEFuDE", "IMFRDE", "RCMDE", "RCMFuDE"
]

METHOD_FAMILIES: dict[str, str] = {
    "MDE": "dispersion",
    "RCMDE": "dispersion",
    "MPE": "permutation",
    "MFE": "fuzzy",
    "MFuDE": "fuzzy_dispersion",
    "TSMEFuDE": "time_shift_ensemble",
    "IMFRDE": "renyi_fuzzy_dispersion",
    "RCMFuDE": "fuzzy_dispersion",
}

# Deterministic feature families that are not entropy jobs.
FUSION_FAMILIES: dict[str, str] = {
    "FUSION_TIME": "time_statistics",
    "FUSION_TIME_SPEC": "time_spectral",
    "FUSION_TIME_ENVELOPE": "time_envelope",
    "FUSION_TIME_ENVELOPE_ENTROPY": "time_envelope_entropy",
    "FUSION_ALL": "all_descriptors",
}

FEATURE_FAMILIES: dict[str, str] = {**METHOD_FAMILIES, **FUSION_FAMILIES}

V1_AUTO_METHODS: tuple[str, ...] = tuple(METHOD_FAMILIES)
