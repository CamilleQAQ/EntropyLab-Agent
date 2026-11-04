"""Small, explicit method catalogue offered to the V1 planner.

This is a policy subset, not a second implementation of the numerical toolbox.
The bundled toolbox owns the eight-method registry and every calculation.
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

V1_AUTO_METHODS: tuple[str, ...] = tuple(METHOD_FAMILIES)
