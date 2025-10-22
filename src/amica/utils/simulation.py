"""Utility functions for simulating data."""
import numpy as np


def generate_toy_data(n_samples=1000, mix_signals=True, noise_factor=None, seed=None):
    """
    Generate toy data consisting of a sine and square wave.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate. Default is 1000.
    mix_signals : bool, optional
        If True, the two signals will be linearly mixed. Default is True.
    noise_factor : float, optional
        If not None, Gaussian noise with this standard deviation will be added to
        the signals (e.g. nois_factor could be set to 0.05).

    Returns
    -------
    mixed_signals : ndarray, shape (n_samples, 2)
        The mixed signals as a 2D numpy array.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1, n_samples + 1)
    a = np.sin(t * 2*np.pi*0.004)
    b = np.sign(np.sin(t * 2*np.pi*0.006))
    if noise_factor is not None:
        a += noise_factor * rng.standard_normal(len(t))
        b += noise_factor * rng.standard_normal(len(t))
    x = np.vstack([a, b]).T

    # optionally mix the signals
    if mix_signals:
        x = np.dot(x, [[0.9, 0.1], [0.1, 0.9]])
    return x
