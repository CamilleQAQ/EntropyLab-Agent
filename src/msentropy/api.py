"""High-level entry layer of the msentropy toolbox.

The package exposes two layers:

- ``msentropy.api`` (this module) is the **high-level API**: one registry
  holds the supported multiscale methods, :func:`compute` normalizes vector orientation, and
  common input mistakes produce readable errors.
- ``msentropy.core.*`` is the **compatibility API**: it exposes the
  validated numerical implementations directly and retains their documented
  MATLAB-compatible edge behavior for scientific reproduction.

For valid inputs, :func:`compute` forwards parameters to the same numerical
functions without changing their calculation.

Example
-------
>>> import numpy as np, msentropy
>>> x = np.random.default_rng(0).standard_normal(2048)
>>> msentropy.compute("MDE", x, m=3, nc=6, tau=1, scale=5).shape
(5,)
>>> msentropy.compute("mde", x, m=3, nc=6, tau=1, scale=5).shape  # case-insensitive
(5,)
>>> msentropy.compute("MDE", x.reshape(-1, 1), m=3, nc=6, tau=1, scale=5).shape  # column OK
(5,)

See ``README.md`` for the method table and the input/output conventions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from msentropy.core.imfrde import imfrde
from msentropy.core.mde import mde
from msentropy.core.mfude import mfude
from msentropy.core.mfe_mu import mfe_mu
from msentropy.core.mpe import mpe
from msentropy.core.rcmde import rcmde
from msentropy.core.rcmfude import rcmfude
from msentropy.core.tsmefude import tsmefude

__all__ = [
    "Method",
    "PARAMETERS",
    "as_signal",
    "compute",
    "get_method",
    "list_methods",
]


@dataclass(frozen=True)
class Method:
    """One registered method as seen by the high-level API.

    Attributes
    ----------
    name : str
        Canonical method name (the paper abbreviation, e.g. ``"RCMFuDE"``).
    full_name : str
        Long English name of the algorithm.
    module : str
        Dotted path of the compatibility module implementing it.
    function : Callable
        The numerical function itself; called with the resolved parameters.
    parameters : tuple of str
        High-level parameter names, in numerical-function signature order.
        The MATLAB argument name may differ (see ``aliases``).
    defaults : dict
        Parameter name -> default value, for parameters the API may omit.
    aliases : dict
        Accepted alternative spelling -> canonical API name (e.g. the
        MATLAB spelling ``t`` for the canonical ``tau``).
    fixed : dict
        Arguments pinned by the API and not exposed, because the reference
        source accepts them without ever reading them.
    """

    name: str
    full_name: str
    module: str
    function: Callable[..., np.ndarray]
    parameters: tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    fixed: Mapping[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        """One-line call signature, e.g. ``MDE(x, m, nc, tau, scale)``."""
        parts = []
        for name in self.parameters:
            if name in self.defaults:
                parts.append(f"{name}={self.defaults[name]!r}")
            else:
                parts.append(name)
        return f"{self.name}(x, {', '.join(parts)})"


#: The eight project-maintained, publicly registered entropy methods.
_METHOD_LIST: tuple[Method, ...] = (
    Method(
        name="MPE",
        full_name="Multiscale Permutation Entropy",
        module="msentropy.core.mpe",
        function=mpe,
        parameters=("m", "tau", "scale"),
        aliases={"t": "tau"},  # MATLAB name of the delay in MPE.m
    ),
    Method(
        name="MDE",
        full_name="Multiscale Dispersion Entropy",
        module="msentropy.core.mde",
        function=mde,
        parameters=("m", "nc", "tau", "scale"),
    ),
    Method(
        name="MFuDE",
        full_name="Multiscale Fuzzy Dispersion Entropy",
        module="msentropy.core.mfude",
        function=mfude,
        parameters=("m", "nc", "tau", "scale", "type_"),
        defaults={"type_": 0},
    ),
    Method(
        name="RCMFuDE",
        full_name="Refined Composite Multiscale Fuzzy Dispersion Entropy",
        module="msentropy.core.rcmfude",
        function=rcmfude,
        parameters=("m", "nc", "tau", "scale", "type_"),
        defaults={"type_": 0},
    ),
    Method(
        name="RCMDE",
        full_name="Refined Composite Multiscale Dispersion Entropy",
        module="msentropy.core.rcmde",
        function=rcmde,
        parameters=("m", "nc", "tau", "scale"),
    ),
    Method(
        name="TSMEFuDE",
        full_name="Time-Shift Multiscale Ensemble Fuzzy Dispersion Entropy",
        module="msentropy.core.tsmefude",
        function=tsmefude,
        parameters=("m", "nc", "tau", "kmax"),
    ),
    Method(
        name="MFE",
        full_name="Multiscale Fuzzy Entropy",
        module="msentropy.core.mfe_mu",
        function=mfe_mu,
        parameters=("m", "r", "n", "tau", "scale"),
    ),
    Method(
        name="IMFRDE",
        full_name="Improved Multiscale Fuzzy Renyi Dispersion Entropy",
        module="msentropy.core.imfrde",
        function=imfrde,
        parameters=("m", "nc", "tau", "scale", "type_", "q"),
        defaults={"type_": 0, "q": 1.3},
    ),
)

#: Canonical name -> :class:`Method`.
_METHODS: dict[str, Method] = {method.name: method for method in _METHOD_LIST}

#: Lower-case canonical name -> canonical name, for case-insensitive lookup.
_LOOKUP: dict[str, str] = {name.lower(): name for name in _METHODS}

#: Canonical method name -> its API parameter names, in signature order.
PARAMETERS: dict[str, tuple[str, ...]] = {
    name: method.parameters for name, method in _METHODS.items()
}


def list_methods() -> tuple[str, ...]:
    """Return the registered multiscale method names.

    Returns
    -------
    tuple of str
        Canonical names, e.g. ``("MPE", "MDE", "MFuDE", ...)``.
    """
    return tuple(_METHODS)


def get_method(method: str) -> Method:
    """Look up a registry entry by name, ignoring case.

    Parameters
    ----------
    method : str
        Canonical name (``"MDE"``), any casing (``"mde"``), or a known
        historical spelling (``"TSMDisE"``).

    Returns
    -------
    Method

    Raises
    ------
    ValueError
        The name is not registered; the message lists every valid name.
    """
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__}; "
            f"valid names: {', '.join(list_methods())}"
        )
    canonical = _LOOKUP.get(method.strip().lower())
    if canonical is None:
        raise ValueError(
            f"unknown method {method!r}; valid names: {', '.join(list_methods())}"
        )
    return _METHODS[canonical]


def as_signal(x: ArrayLike) -> np.ndarray:
    """Normalize input to the toolbox signal convention: 1-D ``float64``.

    A vector-shaped 2-D array ``(1, N)`` or ``(N, 1)`` is raveled to 1-D.
    Therefore a flat array, one row, and one column containing the same
    samples are equivalent in the high-level API. Direct compatibility
    functions may retain orientation-dependent historical behavior.

    Samples are NOT cleaned: NaN samples propagate exactly as in MATLAB.

    Parameters
    ----------
    x : array_like
        Signal samples; a list or array. General matrices (neither
        dimension 1) and scalars are rejected.

    Returns
    -------
    np.ndarray
        Contiguous 1-D ``float64`` array, a view or copy of the input.

    Raises
    ------
    ValueError
        Scalar or non-vector-shaped input.
    TypeError
        Non-real (e.g. complex) or non-numeric samples.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError(
            "signal must be a sequence of samples, got a scalar; "
            "pass a 1-D array, or a (1, N) / (N, 1) array"
        )
    if arr.ndim > 2:
        raise ValueError(
            f"signal must be 1-D or a vector-shaped 2-D array, got shape {arr.shape}"
        )
    if arr.ndim == 2:
        if 1 not in arr.shape:
            raise ValueError(
                f"signal must be 1-D or a vector-shaped 2-D array (1, N) / (N, 1), "
                f"got shape {arr.shape}; flatten it explicitly if that is intended"
            )
        arr = arr.reshape(-1)
    if np.iscomplexobj(arr):
        # numpy would only warn (ComplexWarning) and silently drop the
        # imaginary part; that is never what the caller wants here.
        raise TypeError(
            f"signal samples must be real numbers, got dtype {arr.dtype}"
        )
    try:
        return np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"signal samples must be real numbers, got dtype {arr.dtype}"
        ) from exc


def _resolve_kwargs(method: Method, params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and translate high-level keywords for the numerical function."""
    accepted: dict[str, str] = {}
    for name in method.parameters:
        accepted[name] = name
    for alias, canonical in method.aliases.items():
        accepted.setdefault(alias, canonical)

    resolved: dict[str, Any] = dict(method.fixed)
    seen: dict[str, str] = {}
    for key, value in params.items():
        canonical = accepted.get(key)
        if canonical is None:
            if key in method.fixed:
                raise TypeError(
                    f"{method.name}: the historical reference accepts {key!r} but never "
                    f"reads it (the ZCJ modification made it inert), so the API "
                    f"fixes it at {method.fixed[key]!r} and does not expose it; "
                    f"drop the argument"
                )
            raise TypeError(
                f"{method.name}: unexpected parameter {key!r}; "
                f"accepted parameters: {', '.join(method.parameters)}"
                + (
                    f" (aliases: {', '.join(sorted(method.aliases))})"
                    if method.aliases
                    else ""
                )
            )
        if canonical in seen:
            raise TypeError(
                f"{method.name}: parameter {canonical!r} was passed twice "
                f"(as {seen[canonical]!r} and {key!r})"
            )
        seen[canonical] = key
        resolved[canonical] = value

    missing = [
        name
        for name in method.parameters
        if name not in resolved and name not in method.defaults
    ]
    if missing:
        raise TypeError(
            f"{method.name}: missing required parameter(s): {', '.join(missing)}; "
            f"call signature: {method.signature}"
        )
    for name, value in method.defaults.items():
        resolved.setdefault(name, value)

    _validate_parameter_values(method, resolved)

    # Compatibility functions retain historical argument names. Translate a
    # canonical high-level name (for example ``gamma``) only at this boundary.
    call_names = {
        api_name: historical for historical, api_name in method.aliases.items()
    }
    return {call_names.get(key, key): value for key, value in resolved.items()}


def _validate_parameter_values(
    method: Method, params: Mapping[str, Any]
) -> None:
    """Reject values that are unambiguously invalid for the high-level API."""
    minimums = {
        "m": 2,
        "nc": 2,
        "tau": 1,
        "scale": 1,
        "kmax": 1,
    }
    for name, minimum in minimums.items():
        if name not in params:
            continue
        value = params[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(
                f"{method.name}: {name} must be an integer >= {minimum}, "
                f"got {value!r}"
            )
        if value < minimum:
            raise ValueError(
                f"{method.name}: {name} must be >= {minimum}, got {value!r}"
            )

    for name in ("delta", "gamma"):
        if name not in params:
            continue
        value = params[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{method.name}: {name} must be a finite real number, "
                f"got {value!r}"
            )
        if value < 0:
            raise ValueError(
                f"{method.name}: {name} must be >= 0, got {value!r}"
            )

    if "gamma" in params and params["gamma"] < params["delta"]:
        raise ValueError(
            f"{method.name}: gamma must be >= delta, got "
            f"gamma={params['gamma']!r} and delta={params['delta']!r}"
        )

    for name in ("r", "n"):
        if name not in params:
            continue
        value = params[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{method.name}: {name} must be a finite real number > 0, "
                f"got {value!r}"
            )

    if "type_" in params and params["type_"] not in (0, 1):
        raise ValueError(f"{method.name}: type_ must be 0 or 1")
    if "q" in params:
        q = params["q"]
        if (
            isinstance(q, (bool, np.bool_))
            or not isinstance(q, Real)
            or not np.isfinite(q)
            or q <= 0
            or q == 1
        ):
            raise ValueError(
                f"{method.name}: q must be finite, positive, and different from 1"
            )


def _validate_signal_for_call(
    method: Method, signal: np.ndarray, params: Mapping[str, Any]
) -> None:
    """Apply high-level signal checks without changing compatibility functions."""
    if signal.size == 0:
        raise ValueError(f"{method.name}: signal must contain at least one sample")
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{method.name}: signal contains NaN or infinite samples; "
            "clean, interpolate, or segment the data explicitly before computing"
        )

    maximum = params.get("scale", params.get("kmax"))
    if maximum is None:
        return
    effective_length = (
        (signal.size - int(maximum) + 1) // int(maximum)
        if method.name in {"RCMDE", "RCMFuDE", "IMFRDE"}
        else signal.size // int(maximum)
    )

    if "m" not in params:
        if effective_length < 3:
            raise ValueError(
                f"{method.name}: scale={maximum} leaves approximately "
                f"{effective_length} sample(s) at the largest scale; "
                "Attention Entropy requires at least 3 samples and detectable extrema"
            )
        return

    m = int(params["m"])
    tau = int(params.get("tau", params.get("t", 1)))
    span = (m - 1) * tau
    if effective_length <= span:
        label = "scale" if "scale" in params else "kmax"
        raise ValueError(
            f"{method.name}: {label}={maximum} leaves approximately "
            f"{effective_length} sample(s) at the largest scale, but m={m} "
            f"and tau={tau} require more than {span}; reduce {label}, m, or "
            "tau, or provide a longer signal"
        )


def compute(method: str, x: ArrayLike, **params: Any) -> np.ndarray:
    """Compute a registered multiscale entropy method on a signal.

    Resolves ``method`` in the registry, normalizes ``x`` with
    :func:`as_signal`, checks common input mistakes, and returns the
    numerical function's output as a 1-D ``float64`` array.

    Parameters
    ----------
    method : str
        Canonical method name, case-insensitive: one of :func:`list_methods`.
    x : array_like
        Signal samples, 1-D or vector-shaped 2-D (see :func:`as_signal`).
    **params
        Method parameters as keywords, exactly the ones in ``PARAMETERS``
        for that method. ``type_`` defaults to 0 where it is exposed.

    Returns
    -------
    np.ndarray
        1-D ``float64`` array of entropies: one value per scale (multiscale
        and composite methods) or per time-shift scale (TSM methods).

    Raises
    ------
    ValueError
        Unknown method name, invalid signal shape or samples, an invalid
        parameter value, or insufficient length at the largest scale.
    TypeError
        Unknown or missing parameter, or non-numeric signal samples.

    Examples
    --------
    >>> import numpy as np, msentropy
    >>> x = np.random.default_rng(1).standard_normal(512)
    >>> msentropy.compute("RCMDE", x, m=3, nc=6, tau=1, scale=4).shape
    (4,)
    """
    spec = get_method(method)
    signal = as_signal(x)
    kwargs = _resolve_kwargs(spec, params)
    _validate_signal_for_call(spec, signal, kwargs)
    out = spec.function(signal, **kwargs)
    return np.asarray(out, dtype=np.float64).ravel()
