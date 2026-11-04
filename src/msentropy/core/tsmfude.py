"""Time-Shift Multiscale Fuzzy Dispersion Entropy.

At each scale, TSMFuDE evaluates fuzzy dispersion entropy on every time-shift
phase and averages the phase values."""

import numpy as np

from msentropy.core.decomposition import time_shift_phases
from msentropy.core.fuzzy_disen_ncdf_cumulative import fuzzy_disen_ncdf_cumulative


def tsmfude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    kmax: int,
) -> np.ndarray:
    """Time-Shift Multiscale Fuzzy Dispersion Entropy.

    At each scale, TSMFuDE evaluates fuzzy dispersion entropy on every time-shift
    phase and averages the phase values."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "tsmfude: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1)"
        )

    # MATLAB zeros(1, Kmax) treats a negative dimension as 0
    # and the 1:Kmax loop is then empty, so kmax <= 0 -> empty output.
    out = np.empty(max(kmax, 0))
    for k in range(1, kmax + 1):
        phases = time_shift_phases(x, k)
        en = np.array(
            [fuzzy_disen_ncdf_cumulative(p, m, nc, tau, 0)[0] for p in phases]
        )
        # Seed with the first value so an all-negative-zero phase vector
        # keeps its sign during sequential accumulation.
        total = en[0]
        for val in en[1:]:
            total = total + val
        out[k - 1] = total / k
    return out
