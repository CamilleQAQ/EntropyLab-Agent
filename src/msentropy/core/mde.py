# Adapted from MATLAB code by Hamed Azami and Javier Escudero.
# MATLAB-to-Python translation and modifications by CamilleQAQ.
# SPDX-License-Identifier: CC-BY-4.0

"""Multiscale Dispersion Entropy.

MDE calculates fixed-NCDF dispersion entropy at scale 1 and on each
non-overlapping coarse-grained series at larger scales. The original signal's
mapping statistics are reused across scales."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.dis_en_ncdf_cumulative import dis_en_ncdf_cumulative
from msentropy.core.dis_en_ncdf_ms_cumulative import (
    dis_en_ncdf_ms_cumulative,
)


def mde(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    scale: int,
) -> np.ndarray:
    """Multiscale Dispersion Entropy.

    MDE calculates fixed-NCDF dispersion entropy at scale 1 and on each
    non-overlapping coarse-grained series at larger scales. The original signal's
    mapping statistics are reused across scales."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "mde: x must be 1-D or a (1, N) row; general matrices and "
            "(N, 1) columns are not supported (MATLAB column input hits "
            "an implicit-broadcasting quirk inside the embd2 loop of the "
            "scale-1 call)"
        )
    n = x.size

    # Step 1: original-signal statistics, shared by every scale >= 2.
    # MATLAB's std([])/mean([]) are NaN and std of a single sample is
    # exactly 0, produced silently; numpy warns, so special-case like the
    # NCDF family modules do.
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
    out[0] = dis_en_ncdf_cumulative(x, m, nc, tau, 0)[0]
    for j in range(2, scale + 1):
        xs = coarse_grain(x, j)
        out[j - 1] = dis_en_ncdf_ms_cumulative(
            xs, m, nc, float(mu_x), float(sigma_x), tau, 0
        )[0]
    return out
