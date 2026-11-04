"""Multiscale Fuzzy Entropy with mean coarse-graining.

The signal is standardized once, coarse-grained at each scale, and evaluated
with single-scale fuzzy entropy."""

import numpy as np
from numpy.typing import ArrayLike

from msentropy.core.decomposition import coarse_grain
from msentropy.core.fuzzen import fuzzen
from msentropy.core.mse_mu import _standardize
from msentropy.core.sampen import _as_integral

__all__ = ["mfe_mu"]


def mfe_mu(
    x: ArrayLike,
    m: int,
    r: float,
    n: float,
    tau: int,
    scale: int,
) -> np.ndarray:
    """Multiscale Fuzzy Entropy with mean coarse-graining.

    The signal is standardized once, coarse-grained at each scale, and evaluated
    with single-scale fuzzy entropy."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim > 2 or (x_arr.ndim == 2 and 1 not in x_arr.shape):
        raise ValueError(
            "mfe_mu: x must be 1-D or a vector-shaped 2-D array; "
            "general matrices are not supported"
        )
    scale = _as_integral(scale, "mfe_mu: scale")

    x_flat = x_arr.ravel()

    # The direct function retains its established non-positive-scale shape.
    out = np.full(max(scale, 1), np.nan)

    x_std = _standardize(x_flat)

    # Scale 1 uses the standardized signal without coarse-graining.
    out[0] = fuzzen(x_std, m, r, n, tau)[0]

    for j in range(2, scale + 1):
        out[j - 1] = fuzzen(coarse_grain(x_std, j), m, r, n, tau)[0]
    return out
