"""Multiscale Slope Entropy.

MSlopEn coarse-grains the signal at each scale and calculates entropy from
symbols assigned to consecutive differences by the delta and gama
thresholds."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.slop_entropy import slop_entropy


def mslopen(
    x: np.ndarray,
    m: int,
    delta: float,
    gama: float,
    scale: int,
) -> np.ndarray:
    """Multiscale Slope Entropy.

    MSlopEn coarse-grains the signal at each scale and calculates entropy from
    symbols assigned to consecutive differences by the delta and gama
    thresholds."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim > 2 or (x_arr.ndim == 2 and 1 not in x_arr.shape):
        raise ValueError(
            "mslopen: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1); general matrices are "
            "not supported (MATLAB length() is max(size) and the coarse "
            "graining then linear-indexes the matrix, see "
            "the public input policy)"
        )

    # MATLAB: MSlopEn = [] plus `for j = 1:scale`; a non-positive scale makes
    # the loop empty, so the empty array itself is returned.
    out = np.empty(max(scale, 0))
    for j in range(1, scale + 1):
        # MATLAB: [SlopEn, ~] = SlopEntropy(Xs, m, delta, gama); the pattern
        # histogram is requested and discarded (bit-identical to
        # the single-output call).
        entropy, _ = slop_entropy(coarse_grain(x_arr, j), m, delta, gama)
        out[j - 1] = entropy
    return out
