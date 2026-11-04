"""Signal decomposition primitives shared by multiscale methods.

This module provides non-overlapping coarse-graining, composite offset series,
and time-shift phase construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["coarse_grain", "time_shift_phases"]


def coarse_grain(data: ArrayLike, scale: int) -> np.ndarray:
    """Return non-overlapping block means at one scale."""
    data = np.asarray(data, dtype=np.float64).ravel()
    n = data.size
    j = n // scale  # MATLAB fix(L/S)
    if j == 0:
        raise UnboundLocalError(
            "coarse_grain (MATLAB Multi/Multi_mu): output unassigned, because "
            "J = fix(L/S) = 0 — the MATLAB segment loop never runs when the "
            "input is empty or shorter than the scale factor, so its M_Data "
            "output is never assigned"
        )
    # Sequential per-segment sums: accumulate cannot be blocked, so the
    # last column is bit-identical to a left-to-right loop over each row.
    blocks = data[: j * scale].reshape(j, scale)
    return np.add.accumulate(blocks, axis=1)[:, -1] / scale


def time_shift_phases(signal: ArrayLike, k: int) -> list[np.ndarray]:
    """Return the phase subseries for time-shift scale ``k``."""
    signal = np.asarray(signal, dtype=np.float64).ravel()
    return [signal[b::k] for b in range(k)]
