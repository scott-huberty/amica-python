"""
Test for the AMICA algorithm implementation.

This test runs the main AMICA algorithm and validates that it produces
expected outputs, serving as a regression test during refactoring.
"""
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy import signal

from amica import AMICA, fit_amica
from amica.datasets import data_path
from amica.linalg import pre_whiten
from amica.utils import generate_toy_data, load_fortran_results, load_initial_weights

pytestmark = pytest.mark.timeout(120)


@pytest.mark.slow
@pytest.mark.parametrize(
        "load_weights, n_components, entrypoint",
        [
            (True, None, "function"),
            (True, 16, "function"),
            (False, None, "function"),
            (True, None, "class"),
            (True, 16, "class"),
            ]
        )
def test_eeglab_data(load_weights, n_components, entrypoint):
    """
    Test the complete AMICA algorithm by executing the full main script.

    This test runs the entire algorithm and checks that it completes successfully
    with expected outputs.
    """
    out_type = (
        "amicaout_test"
        if n_components is None
        else "amicaout_dimension_reduction_approx_sphere"
    )
    amica_outdir = data_path() / "eeglab_sample_data" / out_type

    raw = mne.io.read_raw_eeglab(data_path() / "eeglab_sample_data" / "eeglab_data.set")
    dataseg = raw.get_data().T
    dataseg *= 1e6  # Convert from Volts to microVolts

    # Get Fortran results
    want_components = 32 if n_components is None else n_components
    fortran_results = load_fortran_results(
        amica_outdir,
        n_components=want_components,
        n_mixtures=3,
        n_features=32,
        )
    # Load weights
    initial_weights, initial_scales, initial_locations = load_initial_weights(
        amica_outdir,n_components=want_components, n_mixtures=3
    )

    # Run AMICA
    if entrypoint == "function":
        results = fit_amica(
            X=dataseg,
            n_components=n_components,
            max_iter=200,
            tol=1e-7,
            lrate=0.05,
            rholrate=0.05,
            newtrate=1.0,
            w_init=initial_weights if load_weights else None,
            sbeta_init=initial_scales if load_weights else None,
            mu_init=initial_locations if load_weights else None,
            random_state=12345,  # Only used if initial_* are None
            )
        mean = results["mean"]
        S = results["S"]
        W = results["W"]
        A = results["A"]
    elif entrypoint == "class":
        transformer = AMICA(
            n_components=n_components,
            max_iter=200,
            w_init=initial_weights if load_weights else None,
            sbeta_init=initial_scales if load_weights else None,
            mu_init=initial_locations if load_weights else None,
            random_state=12345,  # Only used if initial_* are None
            )
        transformer.fit(dataseg)
        mean = transformer.mean_
        S = transformer.whitening_
        A = S @ transformer.mixing_  # mixing_ is in feature space.
        W = transformer._unmixing[:, :, None] # Expand dims to match Fortran shape

    LL_f = fortran_results["LL"]        # Log-likelihood
    mean_f = fortran_results["mean"]    # Channel means
    S_f = fortran_results["S"]          # Sphering matrix
    A_f = fortran_results["A"]          # Mixing matrix
    W_f = fortran_results["W"]          # Unmixing matrix
    gm_f = fortran_results["gm"]        # Model weights
    c_f = fortran_results["c"]          # Bias term
    alpha_f = fortran_results["alpha"]  # Mixture model parameters
    sbeta_f = fortran_results["sbeta"]  # Scale parameters
    mu_f = fortran_results["mu"]        # Location parameters
    rho_f = fortran_results["rho"]      # Shape parameters

    comp_list_f = fortran_results["comp_list"]
    # Something weird is happening there. I expect (num_comps, num_models) = (32, 1)
    comp_list_f = np.reshape(comp_list_f, (want_components, 2), order="F")

    # These should be equal regardless of initialization
    assert_almost_equal(mean, mean_f)
    assert_almost_equal(S[:want_components], S_f[:want_components])
    # Only accessible when using function entrypoint
    if entrypoint == "function":
        assert_almost_equal(results["c"], c_f)
        assert results["gm"] == gm_f == np.array([1.])

    # The rest depend on initialization
    if load_weights:
        assert_almost_equal(A, A_f, decimal=2)
        assert_almost_equal(W, W_f, decimal=2)
        # The rest are only exposed via function entrypoint
        if entrypoint == "function":
            assert_almost_equal(results["LL"], LL_f, decimal=4)
            assert_allclose(results["LL"], LL_f, atol=1e-4)
            assert_almost_equal(results["alpha"], alpha_f, decimal=2)
            assert_almost_equal(results["sbeta"], sbeta_f, decimal=1)
            assert_almost_equal(results["mu"], mu_f, decimal=0)
            assert_allclose(results["rho"], rho_f, rtol=0, atol=0.02)
    else:
        assert_allclose(A, A_f, atol=0.9)
        assert_allclose(W, W_f, atol=0.9)
        if entrypoint == "function":
            assert_allclose(results["LL"], LL_f, rtol=1e-2)
            assert_allclose(results["alpha"], alpha_f, atol=.4)
            assert_allclose(results["sbeta"], sbeta_f, atol=0.9)
            assert_allclose(results["mu"], mu_f, atol=1.6)
            assert_allclose(results["rho"], rho_f, atol=1)
    # Everything past this point is just figure generation for visual inspection
    if n_components == 16 or entrypoint == "class":
        return

    out_dir = data_path() / "figs"
    out_dir.mkdir(exist_ok=True, parents=True)
    for output in ["python", "fortran"]:
        fig, ax = plt.subplots(
            nrows=8,
            ncols=4,
            figsize=(12, 16),
            constrained_layout=True
            )
        for i, this_ax in zip(range(32), ax.flat):
            mne.viz.plot_topomap(
                results["A"][:, i] if output == "python" else A_f[:, i],
                pos=raw.info,
                axes=this_ax,
                show=False,
            )
            this_ax.set_title(f"Component {i}")
        weights_str = "Using Fortran seed" if load_weights else "Using random seed"
        seed_match = ("_fortran_init" if load_weights else "_random_init")
        if output == "fortran":
            weights_str, seed_match = "", ""
        fig.suptitle(f"AMICA Component Topomaps ({output}) {weights_str}", fontsize=16)
        fig.savefig(out_dir / f"amica_topos_{output}{seed_match}.png")
        plt.close(fig)


    def get_amica_sources(X, W, S, mean):
        """
        Apply AMICA transformation to get ICA sources.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix
        W : ndarray, shape (n_components, n_channels)
            Unmixing matrix from AMICA (for single model, use W[:,:,0])
        S : ndarray, shape (n_channels, n_channels)
            Sphering/whitening matrix
        mean : ndarray, shape (n_channels,)
            Channel means

        Returns
        -------
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
        dataseg, results["W"], results["S"], results["mean"]
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

    threshold = 0.99 if load_weights else 0.6
    assert np.all(np.abs(corrs) > threshold)


    if not load_weights:
        return

    info = mne.create_info(
        ch_names=[f"IC{i}" for i in range(sources_python.shape[1])],
        sfreq=raw.info['sfreq'],
        ch_types='eeg'
    )

    raw_src_python = mne.io.RawArray(sources_python.T, info)
    raw_src_fortran = mne.io.RawArray(sources_fortran.T, info)

    mne.viz.set_browser_backend("matplotlib")
    fig = raw_src_python.plot(scalings=dict(eeg=1))
    fig.savefig(out_dir / f"amica_sources_python_{seed_match}.png")
    plt.close(fig)
    fig = raw_src_fortran.plot(scalings=dict(eeg=1))
    fig.savefig(out_dir / "amica_sources_fortran.png")
    plt.close(fig)


@pytest.mark.parametrize(
        "n_samples, noise_factor, entrypoint",
        [
            (10_000, 0.05, "function"),
            (5_000, None, "function"),
            (10_000, 0.05, "class"),
            (5_000, None, "class"),
            ]
            )
def test_simulated_data(n_samples, noise_factor, entrypoint):
    """Test AMICA on simulated data and compare to Fortran results.

    On this simulated data the convergence between languages varies much more
    and is less stable across runs (at least in the Python version). AFAICT this is
    because of numerical differences that blow up over time, possibly due to denominator
    terms that can get very small. The Fortran version seems more stable in this regard.
    We might need to add some regularization to the updates to make things more stable.
    """
    # Generate toy data
    toy_idx = 1 if n_samples == 10_000 else 2
    x = generate_toy_data(n_samples=n_samples, noise_factor=noise_factor, seed=123456)
    fortran_dir = data_path()
    amicaout_dir = fortran_dir / f"toy_{toy_idx}" / f"amicaout_toy_{toy_idx}"
    weights, scales, locations = load_initial_weights(
        amicaout_dir, n_components=2, n_mixtures=3
        )
    fortran_results = load_fortran_results(
        amicaout_dir, n_components=2, n_mixtures=3
        )
    assert np.all(fortran_results["mean"] == 0)
    # Unpack Fortran results
    A_f = fortran_results["A"]
    W_f = fortran_results["W"]
    alpha_f = fortran_results["alpha"]
    sbeta_f = fortran_results["sbeta"]
    mu_f = fortran_results["mu"]
    rho_f = fortran_results["rho"]
    LL_f = fortran_results["LL"]

    # Run AMICA
    if entrypoint == "function":
        results = fit_amica(
            x,
            mean_center=False,
            whiten=False,
            max_iter=500,
            w_init=weights,
            sbeta_init=scales,
            mu_init=locations,
        )
        S = results["S"]                # Sphering matrix
        mu = results["mu"]              # Location parameters
        rho = results["rho"]            # Shape parameters
        sbeta = results["sbeta"]        # Scale parameters
        W = results["W"]                # Unmixing matrix
        A = results["A"]                # Mixing matrix in space of sphered data
        alpha = results["alpha"]        # Mixture weights
        LL = results["LL"]              # Log-likelihood
        # Compare to Fortran results
        S_f = fortran_results["S"]
        assert_allclose(S, S_f)
    else:
        transformer = AMICA(
            n_components=2,
            whiten=False,
            mean_center=False,
            max_iter=500,
            w_init=weights,
            sbeta_init=scales,
            mu_init=locations,
            )
        transformer.fit(x)
        S = transformer.whitening_
        A = S @ transformer.mixing_  # put mixing_ from feature space to sphered space
        W = transformer._unmixing[:, :, None]  # Expand dims to match Fortran shape

    assert_allclose(A, A_f, rtol=0.1)
    assert_allclose(W, W_f, rtol=0.1)
    # These are only exposed via function entrypoint
    if entrypoint == "function":
        assert_allclose(alpha, alpha_f, rtol=0.7)
        assert_allclose(sbeta, sbeta_f, rtol=0.5)

        # Location and shape parameters are quite unstable across platforms
        want_tol = 0.1 if sys.platform != "win32" else 1.0
        assert_allclose(mu, mu_f, rtol=want_tol)
        assert_allclose(rho, rho_f, rtol=0.5)

    if entrypoint == "function":
         iterations_python = np.count_nonzero(LL)
    else:
        iterations_python = transformer.n_iter_
    iterations_fortran = np.count_nonzero(LL_f)

    # We have to be very lenient here because of the instability across runs..
    # The source of the instability should be investigated further. It might be
    # due to numerical issues in the updates, especially when denominators get small.
    if n_samples == 10_000:
        assert iterations_python < 500
        if entrypoint == "function":
            assert_allclose(LL[:2], LL_f[:2], rtol=.006, atol=1e-7)
            assert_allclose(LL[:10], LL_f[:10], rtol=20, atol=1e-7)
            assert_allclose(LL[:30], LL_f[:30], atol=6)
            assert_allclose(LL[:200], LL_f[:200], atol=6)

    elif n_samples == 5_000:
        # Both programs solved the problem around ~205 iterations
        diff_iters = np.abs(iterations_fortran - iterations_python)
        # On non-Windows we are very close, but Windows takes way longer to converge
        assert diff_iters < 3 if sys.platform != "win32" else diff_iters < 103
        if entrypoint == "function":
            # The first 2 iterations we are very close
            assert_allclose(LL[:2], LL_f[:2])
            # Then we start to diverge a bit...
            assert_allclose(LL[:10], LL_f[:10], 1e-4)
            # By iteration 30 our paths are quite different
            assert_allclose(LL[:30], LL_f[:30], rtol=.1)
            # The end our final log likelihoods have diverged a lot
            assert_allclose(LL[:200], LL_f[:200], rtol=15, atol=5)
            # AFAICT this is because of compounding numerical differences in the updates


def test_reconstruction():
    """Check that the data can be reconstructed from the sources and mixing matrix."""
    # Generate toy data
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # Compute ICA using both function and class entrypoints
    modout = fit_amica(X, n_components=3, random_state=0)
    S = modout["S"] # whitening matrix shape (3, 3)
    A = modout["A"] # mixing matrix (3, 3)
    W = modout["W"][:, :, 0] # unmixing matrix (3, 3)
    mean = modout["mean"] # feature means (3,)

    # We can prove that the ICA model applies by reverting the unmixing.
    # First, using function output directly
    components = W @ S
    stcs = (X - mean) @ components.T
    X_sph_rec = stcs @ A.T
    S_inv_T = np.linalg.pinv(S).T           # or np.linalg.inv(S).T when full-rank
    X_rec = X_sph_rec @ S_inv_T + mean
    np.testing.assert_allclose(X, X_rec)

    # Second, using the class interface
    transformer = AMICA(n_components=3, random_state=0)
    X_new = transformer.fit_transform(X)
    X_rec = transformer.inverse_transform(X_new)
    np.testing.assert_allclose(X, X_rec)


@pytest.mark.parametrize(
        "n_components, do_approx_sphere",
        [(None, True), (16, True), (32, False), (16, False)]
        )
def test_pre_whiten(n_components, do_approx_sphere):
    """Test our Whitening/Sphering implementation against Fortran results."""
    # Load Fortran Directory
    amica_outdir = data_path() / "eeglab_sample_data"

    # Get the Data
    raw = mne.io.read_raw_eeglab(data_path() / "eeglab_sample_data" / "eeglab_data.set")
    dataseg = raw.get_data().T
    dataseg *= 1e6  # Convert from Volts to microVolts

    # Sphere the data
    X_sphered, S, _, _, _ = pre_whiten(
        X=dataseg.copy(),
        n_components=n_components,
        do_approx_sphere=do_approx_sphere,
    )

    # Compare to Fortran results
    if n_components is None:
        if do_approx_sphere:
            sub_dir = "amicaout_test"
        else:
            sub_dir = "amicaout_test_direct_sphere"
        results = load_fortran_results(
            amica_outdir / sub_dir,
            n_components=32,
            n_mixtures=3,
        )
        S_fortran = results["S"]
        np.testing.assert_allclose(S, S_fortran)
    elif n_components == 16:
        # We have two sets of Fortran results to compare against
        if do_approx_sphere:
            sub_dir = "amicaout_dimension_reduction_approx_sphere"
        else:
            sub_dir = "amicaout_dimension_reduction_direct_sphere"
        # Load Fortran results
        results = load_fortran_results(
            amica_outdir / sub_dir,
            n_features=32,
            n_components=16,
            n_mixtures=3
        )
        S_fortran = results["S"]
        # Test
        assert_allclose(np.abs(S), np.abs(S_fortran))

    # Let's actually test the sphered data against some Fortran values we retrieved.
    # FYI The sign of the sphered data between Fortran and Python can be flipped
    # TODO: In the future we could save the actual Fortran sphered data to file
    if n_components == 16 and not do_approx_sphere:
        assert_allclose(abs(X_sphered[0,0]), 1.1875105848378642)
        assert_allclose(abs(X_sphered[1,0]), 0.32247850347596485)
        assert_allclose(abs(X_sphered[19,0]), 2.304779908218825)
        assert_allclose(abs(X_sphered[19,1]), 1.1973479655746415)
        assert_allclose(abs(X_sphered[19,16]), abs(-7.2252664585424204))
    elif n_components == 32 and not do_approx_sphere:
        assert_allclose(abs(X_sphered[0,0]), abs(-1.187510584837864))
        assert_allclose(abs(X_sphered[1,0]), abs(-0.32247850347596457))
        assert_allclose(abs(X_sphered[19,0]), abs(-2.3047799082188241))
        assert_allclose(abs(X_sphered[19,1]), 1.1973479655746426)
        assert_allclose(abs(X_sphered[19,16]), abs(-1.2746894704832441))
    elif n_components == 16 and do_approx_sphere:
        assert_allclose(abs(X_sphered[0,0]), abs(-0.31609955770237225))
        assert_allclose(abs(X_sphered[1,0]), abs(-0.40310811749340258))
        assert_allclose(abs(X_sphered[19,0]), abs(0.056225277237319106))
        assert_allclose(abs(X_sphered[19,1]), abs(0.30019806406010074))
        assert_allclose(X_sphered[19,16], 0.0)
    elif n_components is None and do_approx_sphere:
        assert_allclose(abs(X_sphered[0,0]), abs(-0.18746213684159407))
        assert_allclose(abs(X_sphered[1,0]), abs(-0.15889933957961194))
        assert_allclose(abs(X_sphered[19,0]), abs(0.10283497726419745))
        assert_allclose(abs(X_sphered[19,1]), abs(0.41534937407296713))
        assert_allclose(abs(X_sphered[19,16]), abs(-1.9626236825765133))
    else:
        raise RuntimeError(
            f"Untested combination of n_components={n_components} "
            f"and do_approx_sphere={do_approx_sphere}"
        )

