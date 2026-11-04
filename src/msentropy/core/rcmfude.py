"""Refined Composite Multiscale Fuzzy Dispersion Entropy.

The method combines fuzzy pattern distributions across all composite offsets
before calculating entropy at each scale."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.fuzzy_disen_ncdf_cumulative import (
    fuzzy_disen_ncdf_cumulative,
)
from msentropy.core.fuzzy_disen_ncdf_ms_cumulative import (
    fuzzy_disen_ncdf_ms_cumulative,
)


def rcmfude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    scale: int,
    type_: int,
) -> np.ndarray:
    """Refined Composite Multiscale Fuzzy Dispersion Entropy.

    The method combines fuzzy pattern distributions across all composite offsets
    before calculating entropy at each scale."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "rcmfude: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1)"
        )
    n = x.size
    # MATLAB indexes x(jj:end) element-wise: rows and columns yield the
    # same offset slice, so a flattened view is sufficient.
    x_flat = x.ravel()

    # MATLAB pre-fills NaN*ones(1, Scale) and overwrites scale 1 plus
    # every scale in 2:Scale; for Scale <= 1 only the first entry is
    # written (see Returns).
    out = np.full(max(scale, 1), np.nan)

    # MATLAB evaluates scale 1 BEFORE estimating the statistics, so a
    # failure there (type, NaN samples) fires first, exactly as in the
    # original.
    out[0] = fuzzy_disen_ncdf_cumulative(x, m, nc, tau, type_)[0]

    # Step 3: original-signal statistics, shared by every scale >= 2.
    # MATLAB's std([])/mean([]) are NaN and std of a single sample is
    # exactly 0, produced silently; numpy warns, so special-case like the
    # fuzzy NCDF family modules do.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        mu_x = float(np.mean(x))
        sigma_x = 0.0
    else:
        mu_x = float(np.mean(x))
        sigma_x = float(np.std(x, ddof=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(2, scale + 1):
            pdf_rows = [
                fuzzy_disen_ncdf_ms_cumulative(
                    coarse_grain(x_flat[jj - 1 :], j),
                    m,
                    nc,
                    mu_x,
                    sigma_x,
                    tau,
                    type_,
                )[1]
                for jj in range(1, j + 1)
            ]
            # Plain mean over the offset rows, exactly like MATLAB
            # mean(pdf, 1): sequential accumulation seeded with the first
            # row (no +0 identity) then a division by j, so an all--0
            # column stays negative and NaN propagates.
            pdf = pdf_rows[0].copy()
            for row in pdf_rows[1:]:
                pdf = pdf + row
            pdf = pdf / j
            # NaN != 0 is True here, as in MATLAB (pdf ~= 0).
            p = pdf[pdf != 0.0]
            out[j - 1] = -np.sum(p * np.log(p))
    return out
