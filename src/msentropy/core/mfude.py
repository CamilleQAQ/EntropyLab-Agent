"""Multiscale Fuzzy Dispersion Entropy.

MFuDE evaluates fuzzy dispersion entropy on the signal and on one
non-overlapping coarse-grained series per scale."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.fuzzy_disen_ncdf_cumulative import (
    fuzzy_disen_ncdf_cumulative,
)
from msentropy.core.fuzzy_disen_ncdf_ms_cumulative import (
    fuzzy_disen_ncdf_ms_cumulative,
)


def mfude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    scale: int,
    type_: int,
) -> np.ndarray:
    """Multiscale Fuzzy Dispersion Entropy.

    MFuDE evaluates fuzzy dispersion entropy on the signal and on one
    non-overlapping coarse-grained series per scale."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "mfude: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1); general matrices are "
            "not supported"
        )
    n = x.size

    # Step 1: original-signal statistics, shared by every scale >= 2.
    # MATLAB's std([])/mean([]) are NaN and std of a single sample is
    # exactly 0, produced silently; numpy warns, so special-case like the
    # fuzzy NCDF family modules do.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        mu_x = np.mean(x)
        sigma_x = 0.0
    else:
        mu_x = np.mean(x)
        sigma_x = np.std(x, ddof=1)

    # MATLAB pre-fills NaN*ones(1, Scale) and overwrites scale 1 plus
    # every scale in 2:Scale; for Scale <= 1 only the first entry is
    # written (see Returns).
    out = np.full(max(scale, 1), np.nan)
    out[0] = fuzzy_disen_ncdf_cumulative(x, m, nc, tau, type_)[0]
    for j in range(2, scale + 1):
        xs = coarse_grain(x, j)
        out[j - 1] = fuzzy_disen_ncdf_ms_cumulative(
            xs, m, nc, float(mu_x), float(sigma_x), tau, type_
        )[0]
    return out
