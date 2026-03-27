"""Shared pytest fixtures for AMICA tests."""

import numpy as np
import pytest
from scipy import signal


@pytest.fixture
def sklearn_example_data():
    """Synthetic mixtures used in the scikit-learn FastICA tutorial."""
    rng = np.random.default_rng(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Sinusoidal
    s2 = np.sign(np.sin(3 * time))  # Square wave
    s3 = signal.sawtooth(2 * np.pi * time)  # Sawtooth

    sources = np.c_[s1, s2, s3]
    sources += 0.2 * rng.standard_normal(sources.shape)  # Add noise
    sources /= sources.std(axis=0)  # Standardize

    mixing = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.5, 2.0, 1.0],
            [1.5, 1.0, 2.0],
        ]
    )
    return sources @ mixing.T
