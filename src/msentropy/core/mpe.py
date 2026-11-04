"""Multiscale Permutation Entropy.

MPE evaluates ordinal-pattern entropy on the original signal and on one
non-overlapping coarse-grained series per scale."""

import numpy as np

from msentropy.core.decomposition import coarse_grain
from msentropy.core.permutation_entropy import permutation_entropy


def mpe(
    x: np.ndarray,
    m: int,
    t: int,
    scale: int,
) -> np.ndarray:
    """Multiscale Permutation Entropy.

    MPE evaluates ordinal-pattern entropy on the original signal and on one
    non-overlapping coarse-grained series per scale."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim > 2 or (x_arr.ndim == 2 and 1 not in x_arr.shape):
        raise ValueError(
            "mpe: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1); general matrices are "
            "not supported (MATLAB length() is max(size) and the coarse "
            "graining then linear-indexes the matrix, see "
            "the public input policy)"
        )

    # MATLAB: MPE = [] plus `for j = 1:scale`; a non-positive scale makes
    # the loop empty, so the empty array itself is returned.
    out = np.empty(max(scale, 0))
    for j in range(1, scale + 1):
        out[j - 1] = permutation_entropy(coarse_grain(x_arr, j), m, t)
    return out
