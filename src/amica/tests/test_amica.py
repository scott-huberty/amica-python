"""
Test for the AMICA algorithm implementation.

This test runs the main AMICA algorithm and validates that it produces
expected outputs, serving as a regression test during refactoring.
"""
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import pytest

from amica import fit_amica
from amica.datasets import data_path

pytestmark = pytest.mark.timeout(60)


def load_initial_parameters(fortran_outdir, *, n_components, n_mixtures):
    fortran_outdir = Path(fortran_outdir)
    assert fortran_outdir.exists()

    initial_weights = np.fromfile(
        fortran_outdir / "Wtmp.bin",
        dtype=np.float64
        )
    initial_weights = initial_weights.reshape((n_components, n_components), order="F")
    initial_scales = np.fromfile(
        fortran_outdir / "sbetatmp.bin",
        dtype=np.float64
        )
    initial_scales = initial_scales.reshape((n_mixtures, n_components), order="F")
    initial_scales = initial_scales.T  # Match Our dimension standard
    initial_locations = np.fromfile(
        fortran_outdir / "mutmp.bin",
        dtype=np.float64
        )
    initial_locations = initial_locations.reshape((n_mixtures, n_components), order="F")
    initial_locations = initial_locations.T  # Match Our dimension standard
    return initial_weights, initial_scales, initial_locations


initial_weights, initial_scales, initial_locations = load_initial_parameters(
    data_path() / "amicaout_test", n_components=32, n_mixtures=3
    )

def test_amica_full_algorithm():
    """
    Test the complete AMICA algorithm by executing the full main script.
    
    This test runs the entire algorithm and checks that it completes successfully
    with expected outputs.
    """
    import mne

    raw = mne.io.read_raw_eeglab(data_path() / "eeglab_data.set")
    dataseg = raw.get_data().T
    dataseg *= 1e6  # Convert from Volts to microVolts
    S, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL = fit_amica(
        X=dataseg,
        max_iter=200,
        tol=1e-7,
        lrate=0.05,
        rholrate=0.05,
        newtrate=1.0,
        initial_weights=initial_weights,
        initial_scales=initial_scales,
        initial_locations=initial_locations,
        )
    
    amica_outdir = data_path() / "amicaout_test"
    
    LL_f = np.fromfile(f"{amica_outdir}/LL")
    assert_almost_equal(LL, LL_f, decimal=4)
    assert_allclose(LL, LL_f, atol=1e-4)

    A_f = np.fromfile(f"{amica_outdir}/A")
    A_f = A_f.reshape((32, 32), order="F")
    assert_almost_equal(A, A_f, decimal=2)

    alpha_f = np.fromfile(f"{amica_outdir}/alpha")
    alpha_f = alpha_f.reshape((3, 32), order="F")
    # Remember that alpha (and sbeta, mu etc) are (num_comps, num_mix) in Python
    assert_almost_equal(alpha, alpha_f.T, decimal=2)

    c_f = np.fromfile(f"{amica_outdir}/c")
    c_f = c_f.reshape((32, 1), order="F")
    assert_almost_equal(c, c_f)


    comp_list_f = np.fromfile(f"{amica_outdir}/comp_list", dtype=np.int32)
    # Something weird is happening there. I expect (num_comps, num_models) = (32, 1)
    comp_list_f = np.reshape(comp_list_f, (32, 2), order="F")


    gm_f = np.fromfile(f"{amica_outdir}/gm")
    assert gm == gm_f == np.array([1.])

    mean_f = np.fromfile(f"{amica_outdir}/mean")
    assert_almost_equal(mean, mean_f)

    mu_f = np.fromfile(f"{amica_outdir}/mu", dtype=np.float64)
    mu_f = mu_f.reshape((3, 32), order="F")
    assert_almost_equal(mu, mu_f.T, decimal=0)

    rho_f = np.fromfile(f"{amica_outdir}/rho", dtype=np.float64)
    rho_f = rho_f.reshape((3, 32), order="F")
    assert_almost_equal(rho, rho_f.T, decimal=2)

    S_f = np.fromfile(f"{amica_outdir}/S", dtype=np.float64)
    S_f = S_f.reshape((32, 32,), order="F")
    assert_almost_equal(S, S_f)

    sbeta_f = np.fromfile(f"{amica_outdir}/sbeta", dtype=np.float64)
    sbeta_f = sbeta_f.reshape((3, 32), order="F")
    assert_almost_equal(sbeta, sbeta_f.T, decimal=1)

    W_f = np.fromfile(f"{amica_outdir}/W", dtype=np.float64)
    W_f = W_f.reshape((32, 32, 1), order="F")
    assert_almost_equal(W, W_f, decimal=2)


    for output in ["python", "fortran"]:
        fig, ax = plt.subplots(
            nrows=8,
            ncols=4,
            figsize=(12, 16),
            constrained_layout=True
            )
        for i, this_ax in zip(range(32), ax.flat):
            mne.viz.plot_topomap(
                A[:, i] if output == "python" else A_f[:, i],
                pos=raw.info,
                axes=this_ax,
                show=False,
            )
            this_ax.set_title(f"Component {i}")
        fig.suptitle(f"AMICA Component Topomaps ({output})", fontsize=16)
        fig.savefig(f"/Users/scotterik/devel/projects/amica-python/figs/amica_topos_{output}.png")
        plt.close(fig)


    def get_amica_sources(X, W, S, mean):
        """
        Apply AMICA transformation to get ICA sources.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix
        W : ndarray, shape (n_components, n_channels) 
            Unmixing matrix from AMICA (for single model, use W[:,:,0])
        S : ndarray, shape (n_channels, n_channels)
            Sphering/whitening matrix  
        mean : ndarray, shape (n_channels,)
            Channel means
            
        Returns:
        --------
        sources : ndarray, shape (n_components, n_times)
            Independent component time series
        """
        # 1. Remove mean
        X_centered = X - mean[None, :]

        # 2. Apply sphering
        X_sphered = X_centered @ S

        # 3. Apply ICA unmixing (this is the key step)
        sources = X_sphered @ W[:, :, 0]  # For single model, use W[:,:,0]

        return sources

    sources_python = get_amica_sources(
        dataseg, W, S, mean
    )
    sources_fortran = get_amica_sources(
        dataseg, W_f, S_f, mean_f
    )
    # Now lets check the correlation between the two sources
    # Taking a subset to avoid memory issues
    corrs = np.zeros(sources_python.shape[1])
    for i in range(sources_python.shape[1]):
        corr = np.corrcoef(
            sources_python[::10, i],
            sources_fortran[::10, i]
        )[0, 1]
        corrs[i] = corr
    assert np.all(np.abs(corr) > 0.99)  # Should be very high correlation

    info = mne.create_info(
        ch_names=[f"IC{i}" for i in range(sources_python.shape[1])],
        sfreq=raw.info['sfreq'],
        ch_types='eeg'
    )

    raw_src_python = mne.io.RawArray(sources_python.T, info)
    raw_src_fortran = mne.io.RawArray(sources_fortran.T, info)

    mne.viz.set_browser_backend("matplotlib")
    fig = raw_src_python.plot(scalings=dict(eeg=.3))
    fig.savefig("/Users/scotterik/devel/projects/amica-python/figs/amica_sources_python.png")
    plt.close(fig)
    fig = raw_src_fortran.plot(scalings=dict(eeg=.3))
    fig.savefig("/Users/scotterik/devel/projects/amica-python/figs/amica_sources_fortran.png")
    plt.close(fig)