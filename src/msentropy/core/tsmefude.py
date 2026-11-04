"""Time-Shift Multiscale Ensemble Fuzzy Dispersion Entropy.

At each scale, TSMEFuDE evaluates ensemble fuzzy dispersion entropy on every
time-shift phase and averages the results. It intentionally follows the
published definition; a historical MATLAB helper discarded intermediate
phases because it reset its phase container inside the loop."""

import numpy as np

from msentropy.core.decomposition import time_shift_phases
from msentropy.core.ensfude import ensfude

#: Backwards-compatible non-public alias for the shared phase builder.
_tse_beta = time_shift_phases


def tsmefude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    kmax: int,
) -> np.ndarray:
    """Time-Shift Multiscale Ensemble Fuzzy Dispersion Entropy.

    At each scale, TSMEFuDE evaluates ensemble fuzzy dispersion entropy on every
    time-shift phase and averages the results. It intentionally follows the
    published definition; a historical MATLAB helper discarded intermediate
    phases because it reset its phase container inside the loop."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "tsmefude: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1)"
        )

    # MATLAB zeros(1, Kmax) treats a negative dimension as 0
    # and the 1:Kmax loop is then empty, so kmax <= 0 -> empty output.
    out = np.empty(max(kmax, 0))
    for k in range(1, kmax + 1):
        phases = time_shift_phases(x, k)
        en = np.array([ensfude(p, m, nc, tau)[0] for p in phases])
        # Seed with the first value so an all-negative-zero phase vector
        # keeps its sign during sequential accumulation.
        total = en[0]
        for val in en[1:]:
            total = total + val
        out[k - 1] = total / k
    return out
