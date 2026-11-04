# Adapted from MATLAB code by Hamed Azami and Javier Escudero.
# MATLAB-to-Python translation and modifications by CamilleQAQ.
# SPDX-License-Identifier: CC-BY-4.0

"""Dispersion Entropy with externally supplied NCDF statistics.

This variant uses the provided mean and standard deviation so multiple scales
can share the mapping fitted to the original signal."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.dis_en_ncdf_cumulative import _entropy_from_y


def dis_en_ncdf_ms_cumulative(
    x: np.ndarray,
    m: int,
    nc: int,
    mu: float,
    sigma: float,
    tau: int,
    type_: int,
) -> tuple[float, np.ndarray]:
    """Dispersion Entropy with externally supplied NCDF statistics.

    This variant uses the provided mean and standard deviation so multiple scales
    can share the mapping fitted to the original signal."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "dis_en_ncdf_ms_cumulative: x must be 1-D or a (1, N) row; "
            "general matrices and (N, 1) columns are not supported "
            "(MATLAB column input hits an implicit-broadcasting quirk "
            "inside the embedding loop)"
        )
    mu_a = np.asarray(mu)
    sigma_a = np.asarray(sigma)
    if mu_a.ndim != 0 or sigma_a.ndim != 0:
        raise ValueError(
            "dis_en_ncdf_ms_cumulative: mu and sigma must be scalars "
            "(the historical reference family calls with scalar statistics only)"
        )
    z_all = x.ravel()
    y = toolbox.normcdf(z_all, float(mu_a), float(sigma_a))
    return _entropy_from_y(y, nc, m, tau, type_)
