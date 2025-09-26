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
from amica.utils import generate_toy_data

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


def load_fortran_results(fortran_outdir, *, n_components, n_mixtures):
    """Load results from a completed Fortran AMICA run for comparison."""
    fortran_outdir = Path(fortran_outdir)
    assert fortran_outdir.exists()
    assert fortran_outdir.is_dir()

    # Channel means
    mean_f = np.fromfile(f"{fortran_outdir}/mean")

    # Sphering matrix
    S_f = np.fromfile(f"{fortran_outdir}/S", dtype=np.float64)
    S_f = S_f.reshape((n_components, n_components), order="F")

    # Unmixing matrix
    W_f = np.fromfile(f"{fortran_outdir}/W", dtype=np.float64)
    W_f = W_f.reshape((n_components, n_components, 1), order="F")

    # Mixing matrix
    A_f = np.fromfile(f"{fortran_outdir}/A")
    A_f = A_f.reshape((n_components, n_components), order="F")

    # Bias term
    c_f = np.fromfile(f"{fortran_outdir}/c")
    c_f = c_f.reshape((n_components, 1), order="F")

    # Log-likelihood
    LL_f = np.fromfile(f"{fortran_outdir}/LL")

    # Mixture model parameters
    # Fortran order is n_mixtures x n_components. Ours is n_components x n_mixtures
    alpha_f = np.fromfile(f"{fortran_outdir}/alpha")
    alpha_f = alpha_f.reshape((n_mixtures, n_components), order="F").T
    # Remember that alpha (and sbeta, mu etc) are (num_comps, num_mix) in Python

    # Scale parameters
    sbeta_f = np.fromfile(f"{fortran_outdir}/sbeta", dtype=np.float64)
    sbeta_f = sbeta_f.reshape((n_mixtures, n_components), order="F").T

    # Location parameters
    mu_f = np.fromfile(f"{fortran_outdir}/mu", dtype=np.float64)
    mu_f = mu_f.reshape((n_mixtures, n_components), order="F").T

    rho_f = np.fromfile(f"{fortran_outdir}/rho", dtype=np.float64)
    rho_f = rho_f.reshape((n_mixtures, n_components), order="F").T


    comp_list_f = np.fromfile(f"{fortran_outdir}/comp_list", dtype=np.int32)
    # Something weird is happening there. I expect (num_comps, num_models) = (32, 1)
    comp_list_f = np.reshape(comp_list_f, (n_components, 2), order="F")

    gm_f = np.fromfile(f"{fortran_outdir}/gm")
    return {
        "W": W_f,
        "S": S_f,
        "sbeta": sbeta_f,
        "rho": rho_f,
        "mu": mu_f,
        "mean": mean_f,
        "gm": gm_f,
        "comp_list": comp_list_f,
        "c": c_f,
        "alpha": alpha_f,
        "A": A_f,
        "LL": LL_f
    }



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
    fortran_results = load_fortran_results(amica_outdir, n_components=32, n_mixtures=3)
    LL_f = fortran_results["LL"]
    assert_almost_equal(LL, LL_f, decimal=4)
    assert_allclose(LL, LL_f, atol=1e-4)

    A_f = fortran_results["A"]
    assert_almost_equal(A, A_f, decimal=2)

    alpha_f = fortran_results["alpha"]
    assert_almost_equal(alpha, alpha_f, decimal=2)

    c_f = fortran_results["c"]
    assert_almost_equal(c, c_f)


    comp_list_f = fortran_results["comp_list"]
    # Something weird is happening there. I expect (num_comps, num_models) = (32, 1)
    comp_list_f = np.reshape(comp_list_f, (32, 2), order="F")


    gm_f = fortran_results["gm"]
    assert gm == gm_f == np.array([1.])

    mean_f = fortran_results["mean"]
    assert_almost_equal(mean, mean_f)

    mu_f = fortran_results["mu"]
    assert_almost_equal(mu, mu_f, decimal=0)

    rho_f = fortran_results["rho"]
    assert_almost_equal(rho, rho_f, decimal=2)

    S_f = fortran_results["S"]
    assert_almost_equal(S, S_f)

    sbeta_f = fortran_results["sbeta"]
    assert_almost_equal(sbeta, sbeta_f, decimal=1)

    W_f = fortran_results["W"]
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



@pytest.mark.parametrize("n_samples, noise_factor", [(10_000, 0.05), (5_000, None)])
def test_simulated_data(n_samples, noise_factor):
    """Test AMICA on simulated data and compare to Fortran results.
    

    On this simulated data the convergence between languages varies much more
    and is less stable across runs (at least in the Python version). AFAICT this is
    because of numerical differences that blow up over time, possibly due to denominator
    terms that can get very small. The Fortran version seems more stable in this regard.
    We might need to add some regularization to the updates to make things more stable.
    """
    # Generate toy data
    if n_samples == 5_000:
        pytest.mark.xfail(reason="Not yet working")
    toy_idx = 1 if n_samples == 10_000 else 2
    x = generate_toy_data(n_samples=n_samples, noise_factor=noise_factor)
    fortran_dir = Path("/Users/scotterik/devel/projects/amica-python/amica/tests")
    amicaout_dir = fortran_dir / f"toy_{toy_idx}" / f"amicaout_toy_{toy_idx}"
    weights, scales, locations = load_initial_parameters(
        amicaout_dir, n_components=2, n_mixtures=3
        )
    fortran_results = load_fortran_results(
        amicaout_dir, n_components=2, n_mixtures=3
        )


    S, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL = fit_amica(
    x, centering=False, whiten=False, max_iter=500,
    initial_weights=weights, initial_scales=scales, initial_locations=locations,
    )

    assert np.all(fortran_results["mean"] == 0)

    S_f = fortran_results["S"]
    assert_allclose(S, S_f, atol=.01)

    A_f = fortran_results["A"]
    assert_allclose(A, A_f, rtol=.09)

    W_f = fortran_results["W"]
    assert_allclose(W, W_f, atol=.009)

    alpha_f = fortran_results["alpha"]
    assert_allclose(alpha, alpha_f, rtol=2.0)

    sbeta_f = fortran_results["sbeta"]
    assert_allclose(sbeta, sbeta_f, rtol=3.0)

    mu_f = fortran_results["mu"]
    assert_allclose(mu, mu_f, atol=.15)

    rho_f = fortran_results["rho"]
    assert_allclose(rho, rho_f, atol=1e-7, rtol=.4)


    LL_f = fortran_results["LL"]
    iterations_fortran = np.count_nonzero(LL_f)
    iterations_python = np.count_nonzero(LL)
    if n_samples == 10_000:
        assert iterations_python < 500
        assert_allclose(LL[:2], LL_f[:2], rtol=.006, atol=1e-7)
        assert_allclose(LL[:10], LL_f[:10], rtol=20, atol=1e-7)
        assert_allclose(LL[:30], LL_f[:30], atol=3)
        assert_allclose(LL[:200], LL_f[:200], atol=3)
    
    elif n_samples == 5_000:
        # Both programs solved the problem around ~205 iterations
        assert np.abs(iterations_fortran - iterations_python) < 3
        # The first 2 iterations we are very close
        assert_allclose(LL[:2], LL_f[:2])
        # Then we start to diverge a bit...
        assert_allclose(LL[:10], LL_f[:10], 1e-4)
        # By iteration 30 our paths are quite different
        assert_allclose(LL[:30], LL_f[:30], rtol=.1)
        # The end our final log likelihoods have diverged a lot
        assert_allclose(LL[:200], LL_f[:200], rtol=15, atol=5)
        # AFAICT, this is because of compounding numerical differences in the updates