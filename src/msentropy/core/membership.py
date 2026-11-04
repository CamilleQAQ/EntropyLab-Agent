"""Fuzzy membership functions used by fuzzy dispersion entropy.

The triangular and trapezoidal functions convert class coordinates into
membership weights without performing entropy aggregation."""


def triangle_mf(k: float, z: float) -> float:
    """Triangular membership function used by Fuzzy Dispersion Entropy."""
    if z > k + 1:
        u = 0.0
    elif (z >= k) and (z <= k + 1):
        u = k + 1 - z
    elif (z >= k - 1) and (z <= k):
        u = z - k + 1
    elif z < k - 1:
        u = 0.0
    return u


def trapezoidal_mf(k: float, z: float, nc: float) -> float:
    """Trapezoidal membership function used by Fuzzy Dispersion Entropy."""
    if k == 1:
        if z > 2:
            u = 0.0
        elif (z >= 1) and (z <= 2):
            u = 2 - z
        elif z < 1:
            u = 1.0
    elif k == nc:
        if z > nc:
            u = 1.0
        elif (z >= nc - 1) and (z <= nc):
            u = z - nc + 1
        elif z < nc - 1:
            u = 0.0
    return u
