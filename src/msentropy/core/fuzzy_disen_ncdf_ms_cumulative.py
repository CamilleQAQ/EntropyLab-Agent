"""Fuzzy Dispersion Entropy with external NCDF statistics.

This variant uses a supplied mean and standard deviation so multiple scales
share the original signal's mapping."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.fuzzy_disen import _fuzzy_pdf_entropy_from_z


def fuzzy_disen_ncdf_ms_cumulative(
    x: np.ndarray,
    m: int,
    nc: int,
    mu: float,
    sigma: float,
    tau: int,
    type_: int,
) -> tuple[float, np.ndarray]:
    """Fuzzy Dispersion Entropy with external NCDF statistics.

    This variant uses a supplied mean and standard deviation so multiple scales
    share the original signal's mapping."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "fuzzy_disen_ncdf_ms_cumulative: x must be 1-D (MATLAB row "
            "convention) or a vector-shaped 2-D array (1xN or Nx1); "
            "general matrices are not supported"
        )
    mu_a = np.asarray(mu)
    sigma_a = np.asarray(sigma)
    if mu_a.ndim != 0 or sigma_a.ndim != 0:
        raise ValueError(
            "fuzzy_disen_ncdf_ms_cumulative: mu and sigma must be scalars "
            "(the historical reference family calls with scalar statistics only)"
        )

    # Steps 1-3: NCDF mapping with the EXTERNAL parameters (no sample
    # statistics are estimated here), 1e-10 boundary protection,
    # UNROUNDED class coordinates. The flat index reproduces MATLAB's
    # elementwise normcdf on both row and column x.
    y = toolbox.normcdf(x.ravel(), float(mu_a), float(sigma_a))
    y = np.where(y == 1.0, 1.0 - 1e-10, y)
    y = np.where(y == 0.0, 1e-10, y)
    z = y * nc + 0.5
    return _fuzzy_pdf_entropy_from_z(z, nc, m, tau, type_)
