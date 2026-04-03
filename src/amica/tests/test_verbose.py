import numpy as np
import pytest

from amica._sklearn_interface import AMICA
from amica.utils._verbose import _validate_verbose


@pytest.mark.parametrize(
    "verbose,expected",
    [
        (None, 1),
        (True, 1),
        (False, 0),
        (0, 0),
        (1, 1),
        (2, 2),
    ],
)
def test_verbose_valid(verbose, expected):
    """Check valid inputs return expected outputs."""
    assert _validate_verbose(verbose) == expected


@pytest.mark.parametrize("verbose", [-1, 3, "INFO", 1.5])
def test_verbose_invalid(verbose):
    """Check that invalid inputs raise."""
    with pytest.raises((TypeError, ValueError)):
        _validate_verbose(verbose)


def test_sklearn_verbose(monkeypatch):
    """Check that AMICA Verbose Parameters work."""
    captured = []

    def _fake_fit_amica(X, **kwargs):
        captured.append(kwargs["verbose"])
        n_features = X.shape[1]
        n_components = kwargs["n_components"] or n_features
        return {
            "mean": np.zeros(n_features),
            "S": np.eye(n_features),
            "W": np.eye(n_components),
            "A": np.eye(n_components),
            "LL": np.array([1.0, 0.0]),
            "gm": np.array([1.0]),
            "mu": np.zeros((n_components, kwargs["n_mixtures"])),
            "rho": np.zeros((n_components, kwargs["n_mixtures"])),
            "sbeta": np.ones((n_components, kwargs["n_mixtures"])),
            "c": np.zeros((n_components,)),
            "alpha": np.ones((n_components, kwargs["n_mixtures"])),
        }

    monkeypatch.setattr("amica._sklearn_interface.fit_amica", _fake_fit_amica)
    X = np.random.RandomState(0).randn(10, 3)

    est = AMICA(verbose=2, n_components=3)
    est.fit(X)
    est.fit(X, verbose=0)

    assert captured == [2, 0]
