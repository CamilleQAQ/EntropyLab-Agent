"""Feasibility of an entropy method for a given window length.

A method with parameters ``m``, ``tau`` and a scale factor first coarse-grains
(or time-shifts) the window into series of length ``window // scale``; the pattern
count then needs ``>= (m - 1) * tau + 1`` samples, otherwise the implementation
divides by zero patterns.  Both constraints are enforced by the vendored
``msentropy`` code (``coarse_grain`` raises when ``window // scale == 0`` and the
pattern counter yields ``W == 0``), so this module states them *before* a paid
feature extraction instead of after a failed trial.

The check is dataset-agnostic: it only sees the method name, its parameters and
the window length that the injected ``DatasetSpec`` declares.
"""

from __future__ import annotations

from dataclasses import dataclass

# Methods disagree on the name of their coarsest scale parameter.
SCALE_PARAMETERS = ("scale", "kmax")


@dataclass(frozen=True)
class Feasibility:
    feasible: bool
    required_window: int
    reason: str


def required_window(method: str, parameters: dict) -> int:
    """Smallest window length that can produce at least one pattern."""
    m = int(parameters.get("m", 2))
    tau = int(parameters.get("tau", 1))
    scale = 1
    for key in SCALE_PARAMETERS:
        if key in parameters:
            scale = int(parameters[key])
            break
    per_window = (m - 1) * tau + 1
    return max(scale, 1) * max(per_window, 1)


def check(method: str, parameters: dict, window_size: int) -> Feasibility:
    required = required_window(method, parameters)
    if window_size >= required:
        return Feasibility(True, required, "")
    m = int(parameters.get("m", 2))
    tau = int(parameters.get("tau", 1))
    scale = next(
        (int(parameters[key]) for key in SCALE_PARAMETERS if key in parameters), 1
    )
    return Feasibility(
        False,
        required,
        (
            f"window {window_size} < {required} "
            f"(scale {scale} x ((m-1)*tau+1) = {scale} x {((m - 1) * tau + 1)})"
        ),
    )
