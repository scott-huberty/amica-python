"""Utilities for interfacing with Fortran AMICA outputs."""
from pathlib import Path
import numpy as np


def load_initial_weights(fortran_outdir, *, n_components, n_mixtures):
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