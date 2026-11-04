"""High-level API tests for the eight maintained entropy methods."""

import inspect

import numpy as np
import pytest

import msentropy
from msentropy import api
from msentropy.core.imfrde import imfrde
from msentropy.core.mde import mde
from msentropy.core.mfude import mfude
from msentropy.core.mfe_mu import mfe_mu
from msentropy.core.mpe import mpe
from msentropy.core.rcmde import rcmde
from msentropy.core.rcmfude import rcmfude
from msentropy.core.tsmefude import tsmefude


SIGNAL = np.random.default_rng(20260911).standard_normal(64)
ALL_METHODS = (
    "MPE", "MDE", "MFuDE", "RCMFuDE", "RCMDE", "TSMEFuDE", "MFE", "IMFRDE"
)
CASE_SPECS = {
    "MPE": (dict(m=3, tau=2, scale=3), (3, 2, 3), mpe),
    "MDE": (dict(m=3, nc=6, tau=2, scale=3), (3, 6, 2, 3), mde),
    "MFuDE": (dict(m=3, nc=6, tau=2, scale=3), (3, 6, 2, 3, 0), mfude),
    "RCMFuDE": (dict(m=3, nc=6, tau=2, scale=3), (3, 6, 2, 3, 0), rcmfude),
    "RCMDE": (dict(m=3, nc=6, tau=2, scale=3), (3, 6, 2, 3), rcmde),
    "TSMEFuDE": (dict(m=3, nc=6, tau=2, kmax=3), (3, 6, 2, 3), tsmefude),
    "MFE": (dict(m=2, r=0.2, n=2.0, tau=1, scale=3), (2, 0.2, 2.0, 1, 3), mfe_mu),
    "IMFRDE": (dict(m=2, nc=6, tau=1, scale=3, type_=0, q=1.3), (2, 6, 1, 3, 0, 1.3), imfrde),
}
SWEEP_PARAMS = {
    "MPE": dict(m=3, tau=1, scale=5),
    "MDE": dict(m=3, nc=6, tau=1, scale=5),
    "MFuDE": dict(m=3, nc=6, tau=1, scale=5),
    "RCMFuDE": dict(m=3, nc=6, tau=1, scale=5),
    "RCMDE": dict(m=3, nc=6, tau=1, scale=5),
    "TSMEFuDE": dict(m=3, nc=6, tau=1, kmax=5),
    "MFE": dict(m=2, r=0.2, n=2.0, tau=1, scale=5),
    "IMFRDE": dict(m=2, nc=6, tau=1, scale=5, type_=0, q=1.3),
}
REMOVED_PUBLIC_METHODS = (
    "MSlopEn", "CMFuDE", "MEFuDE", "CMEFuDE", "RCMEFuDE", "TSMPE",
    "TSMSlopEn", "TSMDE", "TSMFuDE", "MSE", "MAttEn", "RCMPE", "RCMSlopEn",
)


def _bit_equal(actual, expected):
    return (
        actual.shape == expected.shape
        and np.array_equal(actual, expected, equal_nan=True)
        and np.array_equal(np.signbit(actual), np.signbit(expected))
    )


def test_registry_contains_exactly_the_eight_maintained_methods():
    assert msentropy.list_methods() == ALL_METHODS
    assert set(api.PARAMETERS) == set(ALL_METHODS)


@pytest.mark.parametrize("name", ALL_METHODS)
def test_parameters_match_the_numerical_signature(name):
    spec = msentropy.get_method(name)
    direct_names = list(inspect.signature(spec.function).parameters)[1:]
    expected = tuple(
        spec.aliases.get(direct_name, direct_name)
        for direct_name in direct_names
        if direct_name not in spec.fixed
    )
    assert spec.parameters == expected


@pytest.mark.parametrize("name", ALL_METHODS)
def test_compute_is_bit_identical_to_the_numerical_function(name):
    kwargs, args, direct = CASE_SPECS[name]
    assert _bit_equal(msentropy.compute(name, SIGNAL, **kwargs), direct(SIGNAL, *args))


@pytest.mark.parametrize("name", ALL_METHODS)
def test_all_methods_return_a_finite_five_scale_curve(name):
    out = msentropy.compute(
        name, np.random.default_rng(7).standard_normal(512), **SWEEP_PARAMS[name]
    )
    assert out.shape == (5,)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", ALL_METHODS)
def test_flat_row_and_column_inputs_are_equivalent(name):
    kwargs = CASE_SPECS[name][0]
    flat = msentropy.compute(name, SIGNAL, **kwargs)
    assert _bit_equal(msentropy.compute(name, SIGNAL.reshape(1, -1), **kwargs), flat)
    assert _bit_equal(msentropy.compute(name, SIGNAL.reshape(-1, 1), **kwargs), flat)


@pytest.mark.parametrize("name", REMOVED_PUBLIC_METHODS)
def test_removed_independent_methods_are_not_publicly_registered(name):
    with pytest.raises(ValueError):
        msentropy.get_method(name)


@pytest.mark.parametrize("name", ["MFuDE", "RCMFuDE"])
def test_fuzzy_dispersion_type_defaults_to_zero(name):
    assert msentropy.get_method(name).defaults == {"type_": 0}


def test_imfrde_defaults_and_rejects_old_spelling():
    assert msentropy.get_method("IMFRDE").defaults == {"type_": 0, "q": 1.3}
    with pytest.raises(ValueError):
        msentropy.get_method("IMFuRDE")


@pytest.mark.parametrize("old_name", ["RCMFDE", "RCMDisEn"])
def test_old_algorithm_spellings_are_not_registered(old_name):
    with pytest.raises(ValueError):
        msentropy.get_method(old_name)


@pytest.mark.parametrize("spelling", ["mde", "MDE", "Mde", "  mde  "])
def test_lookup_ignores_case_and_space(spelling):
    assert msentropy.get_method(spelling).name == "MDE"


def test_mpe_matlab_delay_alias_and_duplicate_rejection():
    assert _bit_equal(
        msentropy.compute("MPE", SIGNAL, m=3, t=2, scale=3),
        msentropy.compute("MPE", SIGNAL, m=3, tau=2, scale=3),
    )
    with pytest.raises(TypeError, match="passed twice"):
        msentropy.compute("MPE", SIGNAL, m=3, t=2, tau=2, scale=3)


@pytest.mark.parametrize(
    ("name", "kwargs", "parameter"),
    [
        ("MPE", dict(m=True, tau=1, scale=3), "m"),
        ("MDE", dict(m=3, nc=1, tau=1, scale=3), "nc"),
        ("MDE", dict(m=3, nc=6, tau=0, scale=3), "tau"),
        ("MDE", dict(m=3, nc=6, tau=1, scale=0), "scale"),
        ("MFE", dict(m=2, r=0.2, n=0.0, tau=1, scale=3), "n"),
        ("IMFRDE", dict(m=2, nc=6, tau=1, scale=3, q=1), "q"),
        ("IMFRDE", dict(m=2, nc=6, tau=1, scale=3, type_=2), "type_"),
    ],
)
def test_invalid_parameter_values_are_rejected(name, kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        msentropy.compute(name, SIGNAL, **kwargs)


def test_missing_and_unexpected_parameters_are_explained():
    with pytest.raises(TypeError, match="missing required parameter.*tau"):
        msentropy.compute("MDE", SIGNAL, m=3, nc=6, scale=3)
    with pytest.raises(TypeError, match="unexpected parameter 'scales'"):
        msentropy.compute("MDE", SIGNAL, m=3, nc=6, tau=1, scales=3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_samples_are_rejected(bad):
    signal = SIGNAL.copy()
    signal[5] = bad
    with pytest.raises(ValueError, match="NaN or infinite"):
        msentropy.compute("MDE", signal, m=3, nc=6, tau=1, scale=3)


def test_shape_empty_length_and_type_safety():
    with pytest.raises(ValueError, match="vector-shaped"):
        msentropy.compute("MDE", np.zeros((2, 3)), m=3, nc=6, tau=1, scale=3)
    with pytest.raises(ValueError, match="at least one sample"):
        msentropy.compute("MDE", [], m=3, nc=6, tau=1, scale=3)
    with pytest.raises(ValueError, match="longer signal"):
        msentropy.compute("MDE", SIGNAL, m=3, nc=6, tau=2, scale=16)
    with pytest.raises(TypeError, match="real numbers"):
        api.as_signal(np.array([1 + 2j]))


def test_input_is_not_mutated():
    original = SIGNAL.copy()
    msentropy.compute("MDE", SIGNAL, m=3, nc=6, tau=1, scale=3)
    assert np.array_equal(SIGNAL, original)


def test_package_exports_target_direct_methods():
    for name in ("compute", "list_methods", "get_method", "as_signal", "PARAMETERS"):
        assert name in msentropy.__all__
    for method_name in ALL_METHODS:
        function = msentropy.get_method(method_name).function
        assert function.__name__ in msentropy.__all__
        assert getattr(msentropy, function.__name__) is function
    assert msentropy.mde is mde
    assert msentropy.imfrde is imfrde
