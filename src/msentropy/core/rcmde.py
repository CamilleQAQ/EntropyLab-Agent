# Adapted from MATLAB code by Hamed Azami and Javier Escudero.
# MATLAB-to-Python translation and modifications by CamilleQAQ.
# SPDX-License-Identifier: CC-BY-4.0

"""Refined Composite Multiscale Dispersion Entropy.

RCMDE pools dispersion-pattern counts across all coarse-graining offsets at a
scale before calculating entropy."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.dis_en_ncdf_cumulative import dis_en_ncdf_cumulative
from msentropy.core.dis_en_ncdf_ms_cumulative import (
    dis_en_ncdf_ms_cumulative,
)


def rcmde(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    scale: int,
) -> np.ndarray:
    """Refined Composite Multiscale Dispersion Entropy.

    RCMDE pools dispersion-pattern counts across all coarse-graining offsets at a
    scale before calculating entropy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "rcmde: x must be 1-D or a (1, N) row; general matrices and "
            "(N, 1) columns are not supported (scale 1 calls the DisEn "
            "leaf directly and MATLAB's column input hits an implicit-"
            "broadcasting behavior inside its embedding loop)"
        )
    n = x.size
    # MATLAB indexes x(jj:end) element-wise, so rows and columns yield the
    # same offset slice; a flattened view is sufficient for the coarse
    # graining and for the statistics.
    x_flat = x.ravel()

    # MATLAB pre-fills NaN*ones(1, Scale) and overwrites scale 1 plus
    # every scale in 2:Scale; for Scale <= 1 only the first entry is
    # written (see Returns).
    out = np.full(max(scale, 1), np.nan)

    # MATLAB evaluates scale 1 BEFORE estimating the statistics. type = 0
    # is hard-wired: the original has no `type` argument (an extra one
    # dies on MATLAB:TooManyInputs), and the nested leaf inlines
    # the `case 0` body of the standalone file.
    out[0] = dis_en_ncdf_cumulative(x, m, nc, tau, 0)[0]

    # Step 3: original-signal statistics, shared by every scale >= 2.
    # MATLAB's std([])/mean([]) are NaN and std of a single sample is
    # exactly 0, produced silently; numpy warns, so special-case them.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        mu_x = float(np.mean(x_flat))
        sigma_x = 0.0
    else:
        mu_x = float(np.mean(x_flat))
        sigma_x = float(np.std(x_flat, ddof=1))

    for j in range(2, scale + 1):
        pdf_rows = [
            dis_en_ncdf_ms_cumulative(
                coarse_grain(x_flat[jj - 1 :], j), m, nc, mu_x, sigma_x, tau, 0
            )[1]
            for jj in range(1, j + 1)
        ]
        # Plain mean over the offset rows, exactly like MATLAB
        # mean(pdf, 1): sequential accumulation seeded with the first row
        # (no +0 identity) then a division by j, so an all--0 column stays
        # negative and NaN propagates through the whole column.
        pdf = pdf_rows[0].copy()
        for row in pdf_rows[1:]:
            pdf = pdf + row
        pdf = pdf / j
        # NaN != 0 is True here, as in MATLAB (pdf ~= 0).
        p = pdf[pdf != 0.0]
        out[j - 1] = -np.sum(p * np.log(p))
    return out
