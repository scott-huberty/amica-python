"""Utilities for interfacing with Fortran AMICA outputs."""
import inspect
from dataclasses import MISSING, asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import numpy as np


def load_initial_weights(fortran_outdir, *, n_components, n_mixtures):
    """Load w_init, sbeta_init, and mu_init binary files from a Fortran AMICA run."""
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


def load_fortran_results(fortran_outdir, *, n_components, n_mixtures, n_features=None):
    """Load results from a completed Fortran AMICA run for comparison."""
    fortran_outdir = Path(fortran_outdir)
    assert fortran_outdir.exists()
    assert fortran_outdir.is_dir()
    if n_features is None:
        n_features = n_components

    # Channel means
    mean_f = np.fromfile(f"{fortran_outdir}/mean")

    # Sphering matrix
    S_f = np.fromfile(f"{fortran_outdir}/S", dtype=np.float64)
    S_f = S_f.reshape((n_features, n_features), order="F")

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


def write_data(data, filename):
    """Save data to a binary file in Fortran-compatible format.

    Parameters
    ----------
    data : array-like
        The data of shape (n_samples, n_features) to save. Will be converted to a
        Fortran-contiguous array of type float32.
    filename : str or Path
        The path to the output binary file.

    Returns
    -------
    data : np.ndarray
        The Fortran-contiguous array that was saved.
    path : Path
        The path to the saved file.
    """
    # tofile ravels matrices in C order, so force Fortran order.
    fpath = Path(filename).expanduser().resolve()
    # We actually have to write in C order to be Fortran compatible.
    # Or transpose the data First and write in Fortran order.
    # Because Fortran program expects (n_features, n_samples)
    vector = data.T.astype("<f4").ravel(order="F")
    vector.tofile(fpath)
    return fpath, data


def load_data(filename, *, dtype=np.float32, shape=None):
    """Load binary data file that saved for use with Fortran AMICA.

    Parameters
    ----------
    filename : str or Path
        The path to the input binary file.
    dtype : data-type
        The desired data-type for the loaded array. Default is np.float32.
    shape : tuple of int
        The shape of the array to load.

    Returns
    -------
    data : np.ndarray
        The Fortran-contiguous array that was loaded.

    Notes
    -----
    Fortran stores arrays in column-major order, and the Fortran program
    expectes data in shape (n_features, n_samples). So when loading data
    for use in Python, you should reshape to (n_features, n_samples) and
    then transpose to (n_samples, n_features) to match the common Python
    convention.

    Examples
    --------
    >>> data = load_data('data.bin', shape=(64, 1000)).T
    """
    data = np.fromfile(filename, dtype=dtype)
    if shape is not None:
        data = data.reshape(shape, order="F")
    return data


@lru_cache(maxsize=1)
def _get_fortran_param_defaults():
    """Collect Fortran parameter defaults aligned with AMICA-Python."""
    defaults = {}
    for field in fields(FortranParams):
        if field.default is not MISSING:
            defaults[field.name] = field.default

    # Public API defaults from fit_amica.
    from amica.core import fit_amica

    fit_defaults = {
        name: param.default
        for name, param in inspect.signature(fit_amica).parameters.items()
        if param.default is not inspect.Signature.empty
    }
    fit_to_fortran = {
        "max_iter": "max_iter",
        "n_models": "num_models",
        "n_mixtures": "num_mix_comps",
        "lrate": "lrate",
        "rholrate": "rholrate",
        "pdftype": "pdftype",
        "do_newton": "do_newton",
        "newt_start": "newt_start",
        "newtrate": "newtrate",
        "newt_ramp": "newt_ramp",
        "do_reject": "do_reject",
    }
    for fit_key, fortran_key in fit_to_fortran.items():
        if fit_key in fit_defaults:
            defaults[fortran_key] = fit_defaults[fit_key]

    # Internal defaults from constants.py.
    from amica import constants as c

    const_to_fortran = {
        "lratefact": "lratefact",
        "rholratefact": "rholratefact",
        "use_min_dll": "use_min_dll",
        "use_grad_norm": "use_grad_norm",
        "min_dll": "min_dll",
        "min_nd": "min_grad_norm",
        "do_opt_block": "do_opt_block",
        "mineig": "mineig",
        "minlrate": "minlrate",
        "rho0": "rho0",
        "minrho": "minrho",
        "maxrho": "maxrho",
        "invsigmax": "invsigmax",
        "invsigmin": "invsigmin",
        "dorho": "do_rho",
        "doscaling": "doscaling",
        "maxdecs": "max_decs",
        "share_comps": "share_comps",
        "share_start": "share_start",
        "share_iter": "share_iter",
        "outstep": "writestep",
        "fix_init": "fix_init",
    }
    for const_key, fortran_key in const_to_fortran.items():
        if hasattr(c, const_key):
            defaults[fortran_key] = getattr(c, const_key)

    return defaults


def write_param_file(fpath, data, **kwargs):
    """Write a Fortran AMICA parameter file.

    Parameters
    ----------
    fpath : str or Path
        The path to the output parameter file.
    data : np.ndarray
        The data array to write to the file.
    **kwargs : dict
        Additional parameters to write to the file. ``files`` and ``outdir`` are
        required and should be passed via kwargs.

    Returns
    -------
    path : Path
        The path to the saved parameter file.
    """
    fpath = Path(fpath).expanduser().resolve()
    missing = [key for key in ("files", "outdir") if key not in kwargs]
    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(
            f"Missing required parameter(s): {missing_joined}. Pass them via kwargs."
        )

    merged = {**_get_fortran_param_defaults(), **kwargs}
    merged.setdefault("data_dim", data.shape[1])
    merged.setdefault("field_dim", data.shape[0])
    merged.setdefault("block_size", data.shape[0])
    merged.setdefault("pcakeep", data.shape[1])

    params = FortranParams(**merged)
    params_dict = params.to_param_dict()

    fpath.write_text("".join(f"{key} {value}\n" for key, value in params_dict.items()))
    return fpath, params


@dataclass
class FortranParams:
    """Dataclass to hold Fortran AMICA parameters."""

    # Required parameters
    files:          str | Path
    outdir:         str | Path
    # Data Shape
    block_size:     int
    data_dim:       int  # n_features
    field_dim:      int # n_samples
    max_iter:       int = 500
    blk_min:        int | None = None
    blk_step:       int | None = None
    blk_max:        int | None = None
    # Whitening
    do_mean:        int = 1
    do_sphere:      int = 1
    doPCA:          int = 1
    pcakeep:        int | None = None
    pcadb:          float = 30.000000
    # You'll probably never need to change these...
    # Main Model Params
    num_models :    int = 1
    max_threads :   int = 1  # Single-threaded (aids debugging)
    # Newton
    do_newton:      int =1
    newt_start:     int = 50
    newt_ramp:      int = 10
    newtrate:       float = 1.000000
    # Learning Rates
    lrate:          float = 0.050000
    rholrate:       float = 0.050000
    lratefact:      float = 0.500000
    rholratefact:   float = 0.500000
    # Convergence
    use_min_dll:    int | bool = 1
    min_dll:        float = 1.000000e-09
    use_grad_norm:  int = 1
    min_grad_norm:  float = 1.000000e-07
    # Misc.
    do_opt_block:   int | bool = 0
    num_mix_comps:  int = 3
    pdftype:        int = 0
    num_samples:    int = 1
    field_blocksize: int = 1
    do_history:     int = 0
    histstep:       int = 10
    share_comps:    int = 0
    share_start:     int = 100
    comp_thresh:    float = 0.990000
    share_iter:     int = 100
    minlrate:       float = 1.000000e-08
    mineig:         float = 1.000000e-12
    rho0:           float = 1.500000
    minrho:         float = 1.000000
    maxrho:         float = 2.000000
    kurt_start:     int = 3
    num_kurt:       int = 5
    kurt_int:       int = 1
    # Rejection
    do_reject:      int = 0
    numrej:         int = 3
    rejsig:         float = 3.000000
    rejstart:       int = 2
    rejint:         int = 3
    # Saving
    writestep:      int = 1  # Write every iteration (aids debugging)
    write_nd:       int = 0
    write_LLt:      int = 1
    decwindow:      int = 1
    max_decs:       int = 3
    fix_init:       int = 0
    update_A:       int = 1
    update_c:       int = 1
    update_gm:      int = 1
    update_alpha:   int = 1
    update_mu:      int = 1
    update_beta:    int = 1
    invsigmax:      float = 100.000000
    invsigmin:      float = 1.0e-08
    do_rho:         int = 1
    # Debugging
    load_rej:       int = 0
    load_W:         int = 0
    load_c:         int = 0
    load_gm:        int = 0
    load_alpha:     int = 0
    load_mu:        int = 0
    load_beta:      int = 0
    load_rho:       int = 0
    load_comp_list: int = 0
    byte_size:      int = 4
    doscaling:      int = 1
    scalestep:      int = 1

    def __post_init__(self):
        """Initialize attributes."""
        if self.blk_min is None:
            self.blk_min = self.block_size // 4
        if self.blk_step is None:
            self.blk_step = self.block_size // 4
        if self.blk_max is None:
            self.blk_max = self.block_size
        if self.pcakeep is None:
            self.pcakeep = self.data_dim

        # Convert bools to int
        for field in fields(self):
            if isinstance(getattr(self, field.name), bool):
                setattr(self, field.name, int(getattr(self, field.name)))

    def to_param_dict(self):
        """Convert to a dict suitable for writing to aFortran AMICA parameter file."""
        return asdict(self)
