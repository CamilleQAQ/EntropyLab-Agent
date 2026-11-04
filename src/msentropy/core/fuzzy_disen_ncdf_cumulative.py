"""Fuzzy Dispersion Entropy with a fixed NCDF mapping.

Samples receive fuzzy memberships in NCDF-derived classes; embedded fuzzy
patterns are then summarized with Shannon entropy."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.fuzzy_disen import _fuzzy_pdf_entropy_from_z


def fuzzy_disen_ncdf_cumulative(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    type_: int,
) -> tuple[float, np.ndarray]:
    """Fuzzy Dispersion Entropy with a fixed NCDF mapping.

    Samples receive fuzzy memberships in NCDF-derived classes; embedded fuzzy
    patterns are then summarized with Shannon entropy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "fuzzy_disen_ncdf_cumulative: x must be 1-D (MATLAB row "
            "convention) or a vector-shaped 2-D array (1xN or Nx1); "
            "general matrices are not supported"
        )
    n = x.size

    # MATLAB's mean([]) / std([]) are NaN, produced silently; numpy
    # warns on empty input, so handle the empty case explicitly.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        # MATLAB's std of a single sample is exactly 0, while
        # np.std(ddof=1) gives NaN and warns. With sigma = 0 the NCDF
        # mapping takes its Heaviside path exactly like MATLAB.
        mu_x = np.mean(x)
        sigma_x = 0.0
    else:
        mu_x = np.mean(x)
        sigma_x = np.std(x, ddof=1)

    # Step 1-3: NCDF mapping, 1e-10 boundary protection, UNROUNDED class
    # coordinates. The boundary protection also covers sigma = 0.
    y = toolbox.normcdf(x, mu_x, sigma_x)
    y = np.where(y == 1.0, 1.0 - 1e-10, y)
    y = np.where(y == 0.0, 1e-10, y)
    z = y * nc + 0.5
    return _fuzzy_pdf_entropy_from_z(z, nc, m, tau, type_)
