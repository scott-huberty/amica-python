"""Module containing amica funciton entry point."""

import time
from dataclasses import dataclass
from warnings import warn

import torch
from sklearn.exceptions import ConvergenceWarning

from amica._types import (
    DataTensor2D,
)
from amica.constants import (
    doscaling,
    epsdble,
    invsigmax,
    invsigmin,
    lratefact,
    maxdecs,
    maxincs,
    maxrho,
    mineig,
    minlog,
    minlrate,
    minrho,
    outstep,
    rholratefact,
    share_comps,
    share_iter,
    share_start,
    use_grad_norm,
    use_min_dll,
)
from amica.kernels import (
    accumulate_alpha_stats,
    accumulate_beta_stats,
    accumulate_c_stats,
    accumulate_kappa_stats,
    accumulate_lambda_stats,
    accumulate_mu_stats,
    accumulate_rho_stats,
    accumulate_sigma2_stats,
    compute_loglikelihood_and_responsibilities,
    compute_model_loglikelihood_sum,
    compute_preactivations,
    compute_scaled_scores,
    compute_source_densities,
    compute_source_scores,
    compute_weighted_responsibilities,
    precompute_weighted_scores,
)
from amica.linalg import (
    compute_sign_log_determinant,
    get_initial_model_log_likelihood,
    get_unmixing_matrices,
    pre_whiten,
)
from amica.optim import (
    AndersonEMAccelerator,
    pack_state,
    unpack_state,
)
from amica.optim.anderson import is_valid_state
from amica.state import (
    AmicaAccumulators,
    AmicaConfig,
    AmicaState,
    IterationMetrics,
    get_initial_state,
    initialize_accumulators,
)

from ._batching import BatchLoader, choose_batch_size
from ._newton import compute_newton_terms
from .utils._logging import _emit_status, log, set_log_level
from .utils._progress import make_progress_bar
from .utils._verbose import _validate_verbose


DEFAULT_OPTIMIZER_KWARGS: dict[str, object] = {
    "accelerator_order": 1,
    "accelerator_damping": 1.0,
    "accelerator_ridge": 1e-8,
    "accelerator_eps_monotone": 0.0,
    "accelerator_start_iter": 5,
    "accelerator_period": 1,
    "accelerator_max_restarts": 20,
    "accelerator_validate_candidate": True,
    "accelerator_daarem_alpha": 1.2,
    "accelerator_daarem_kappa": 25,
    "accelerator_cycl_monotone_tol": 0.0,
}


def _resolve_optimizer_kwargs(
        optimizer_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    resolved = DEFAULT_OPTIMIZER_KWARGS.copy()
    if optimizer_kwargs is not None:
        unknown = set(optimizer_kwargs) - set(resolved)
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown optimizer_kwargs keys: {unknown_str}")
        resolved.update(optimizer_kwargs)
    return resolved


@dataclass(slots=True)
class EMStepResult:
    """Result of one plain AMICA outer EM iteration."""

    state: AmicaState
    wc: torch.Tensor
    likelihood: torch.Tensor
    ndtmpsum: torch.Tensor
    lrate: float
    rholrate: float


@dataclass(slots=True)
class AccelerationOutcome:
    """Bookkeeping for optional post-EM acceleration."""

    accepted: bool = False
    attempted: bool = False
    reason: str = "disabled"
    history: int = 0
    restart: bool = False
    candidate_loglik: float | None = None


def _ensure_batch_scratch(
        accumulators: AmicaAccumulators,
        *,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
        device: torch.device | str,
) -> None:
    """Allocate reusable per-batch scratch buffers when shape changes."""
    if (
            accumulators.scratch_y is None
            or accumulators.scratch_z is None
            or accumulators.scratch_fp is None
            or accumulators.scratch_ufp is None
            or accumulators.scratch_y.shape != shape
            or accumulators.scratch_z.shape != shape
            or accumulators.scratch_fp.shape != shape
            or accumulators.scratch_ufp.shape != shape
    ):
        accumulators.scratch_y = torch.empty(shape, dtype=dtype, device=device)
        accumulators.scratch_z = torch.empty(shape, dtype=dtype, device=device)
        accumulators.scratch_fp = torch.empty(shape, dtype=dtype, device=device)
        accumulators.scratch_ufp = torch.empty(shape, dtype=dtype, device=device)


def fit_amica(
        X,
        *,
        whiten="zca",
        mean_center=True,
        n_components=None,
        device="cpu",
        n_mixtures=3,
        max_iter=500,
        tol=1e-7,
        lrate=0.05,
        rholrate=0.05,
        pdftype=0,
        do_newton=True,
        newt_start=50,
        newtrate=1.0,
        newt_ramp=10,
        batch_size=None,
        w_init=None,
        sbeta_init=None,
        mu_init=None,
        do_reject=False,
        optimizer="em",
        optimizer_kwargs=None,
        random_state=None,
        verbose=1,
):
    """Perform Adaptive Mixture Independent Component Analysis (AMICA).

    Implements the AMICA algorithm as described in :footcite:t:`palmer2012` and
    :footcite:t:`palmer2008`, and originally implemented in :footcite:t:`amica`.

    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        Training data, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
    n_components : int, optional
        Number of components to extract. If ``None`` (default), set to ``n_features``.
        Note that the number of components may be reduced during whitening if the data
        are rank-deficient.
    n_mixtures: int, optional, default=3
         Number of mixtures components to use in the Gaussian Mixture Model (GMM) for
         each component's source density. default is ``3``.
    batch_size : int, optional
        Batch size for processing data in chunks along the samples axis. If ``None``,
        the batch size is chosen automatically to keep peak memory under ~1.5 GB, and
        warns if the batch size is below ~8k samples. If the input data is small enough
        to process in one shot, no batching is used. If you want to enforce no
        batching, you can override this memory cap by setting batch_size explicitly,
        e.g. to  `X.shape[0]` to process all samples at once. but note that this may
        lead to high memory usage for large datasets.
    device : str, optional
        Device to run the computations on. Can be either 'cpu' or 'cuda' for GPU
        acceleration. Note that using 'cuda' requires a compatible NVIDIA GPU and
        the appropriate CUDA drivers installed.
    whiten : str {"zca", "pca", "variance"}
        Whitening method to apply to the data before fitting AMICA. Options are:
        - "zca": Zero-phase component analysis (ZCA) whitening.
        - "pca": Principal component analysis (PCA) whitening.
        - "variance": Only variance normalization of the features is done (no sphering).
    mean_center : bool, optional
        If ``True``, X is mean corrected.
    max_iter : int, optional
        Maximum number of iterations to perform. Default is ``500``.
    random_state : int or None, optional (default=None)
        Used to perform a random initialization when w_init is not provided.
        If int, random_state is the seed used by the random number generator during
        whitening, and is used to set the seed during optimization initialization.
    w_init : array-like, shape (``n_components``, ``n_components``), optional
        Initial weights for the mixture components. If None, weights are initialized
        randomly. This is meant to be used for testing and debugging purposes only.
    sbeta_init : array-like, shape (``n_components``, ``n_mixtures``), optional
        Initial scales (sbeta) for the mixture components. If None, scales are
        initialized randomly. This is meant to be used for testing and debugging
        purposes only.
    mu_init : array-like, shape (``n_components``, ``n_mixtures``), optional
        Initial locations (mu) for the mixture components. If None, locations are
        initialized randomly. This is meant to be used for testing and debugging
        purposes only.
    lrate : float, default=0.05
        Initial learning rate for the natural gradient.
    rholrate : float = default=0.05
        initial learning rate for shape parameters.
    pdftype : int, default=0
        Type of source density model to use. Currently only ``0`` is supported,
        which corresponds to the Gaussian Mixture Model (GMM) density.
    do_newton : bool, default=True
        If ``True``, the optimization method will switch from Stochastic Gradient
        Descent (SGD) to newton updates after ``newt_start`` iterations. If ``False``,
        only SGD updates are used.
    newt_start : int, default=50
        Number of iterations before switching to Newton updates if ``do_newton`` is
        ``True``.
    newtrate : float, default=1.0
        learning rate for newton iterations.
    optimizer : {"em", "anderson", "daarem"}, default="em"
        Outer-loop optimizer / acceleration path.
    optimizer_kwargs : dict or None, default=None
        Optional accelerator settings for ``optimizer="anderson"`` or
        ``optimizer="daarem"``. Supported keys are ``accelerator_order``,
        ``accelerator_damping``, ``accelerator_ridge``,
        ``accelerator_eps_monotone``, ``accelerator_start_iter``,
        ``accelerator_period``, ``accelerator_max_restarts``,
        ``accelerator_validate_candidate``, ``accelerator_daarem_alpha``,
        ``accelerator_daarem_kappa``, and
        ``accelerator_cycl_monotone_tol``. If ``None``, AMICA uses the default
        accelerator settings.
    verbose : int, default=1
        Output mode during optimization:

        - ``0``: silent
        - ``1``: progress bar
        - ``2``: per-iteration FORTRAN-style logs

    Returns
    -------
    results : dict
        Dictionary containing the following entries:

        - mean : array, shape (``n_features``,) | ``None``
            The mean over features. if ``do_mean=False``, this is ``None``.
        - S : array, shape (``n_components``, ``n_features``)
            The sphering (whitening) matrix applied to the data.
        - W : array, shape (``n_components``, ``n_components``)
            The unmixing matrix.
        - A : array, shape (``n_components``, ``n_components``)
            The mixing matrix in the space of sphered data. To get the mixing matrix
            in the original data space, use ``np.linalg.pinv(S) @ A``.
        - LL : array, shape (``max_iter``,)
            The log-likelihood values at each iteration. If the algorithm converged
            before reaching ``max_iter``, the remaining entries will be zero.
        - gm : array, shape (1,)
            The Gaussian mixture model weights. Since only one model is supported,
            this will be of shape (1,).
        - mu : array, shape (``n_components``, ``n_mixtures``)
            The location parameters for the mixture components, i.e. the means of the
            mixture components.
        - rho : array, shape (``n_components``, ``n_mixtures``)
            The shape parameters for the mixture components.
        - sbeta : array, shape (``n_components``, ``n_mixtures``)
            The scale (precision) parameters for the mixture components.
        - alpha : array, shape (``n_components``, ``n_mixtures``)
            The mixture weights for the mixture components.
        - c : array, shape (``n_components``,)
            The model bias terms.

    Notes
    -----
    In Fortran AMICA, ``alpha``, ``sbeta``, ``mu``, and ``rho`` are of shape
    (``n_mixtures``, ``n_components``) (transposed compared to here).

    References
    ----------
    .. footbibliography::

    """
    verbose = _validate_verbose(verbose)
    set_log_level("INFO" if verbose == 2 else "ERROR")
    optimizer_kwargs = _resolve_optimizer_kwargs(optimizer_kwargs)

    if batch_size is None:
        batch_size = choose_batch_size(
            N=X.shape[0],
            n_comps=n_components if n_components is not None else X.shape[1],
            n_mix=n_mixtures,
        )
    # Step 1: Create config and state objects (new dataclass approach)
    config = AmicaConfig(
        n_features=X.shape[1],  # Number of channels (corrected from X.shape[1])
        n_components=n_components if n_components is not None else X.shape[1],
        n_models=1,
        n_mixtures=n_mixtures,
        max_iter=max_iter,
        batch_size=batch_size,
        device=torch.device(device),
        pdftype=pdftype,
        tol=tol,
        lrate=lrate,
        rholrate=rholrate,
        do_newton=do_newton,
        newt_start=newt_start,
        newtrate=newtrate,
        newt_ramp=newt_ramp,
        do_reject=do_reject,
        optimizer=optimizer,
        accelerator_order=int(optimizer_kwargs["accelerator_order"]),
        accelerator_damping=float(optimizer_kwargs["accelerator_damping"]),
        accelerator_ridge=float(optimizer_kwargs["accelerator_ridge"]),
        accelerator_eps_monotone=float(optimizer_kwargs["accelerator_eps_monotone"]),
        accelerator_start_iter=int(optimizer_kwargs["accelerator_start_iter"]),
        accelerator_period=int(optimizer_kwargs["accelerator_period"]),
        accelerator_max_restarts=int(optimizer_kwargs["accelerator_max_restarts"]),
        accelerator_validate_candidate=bool(
            optimizer_kwargs["accelerator_validate_candidate"]
        ),
        accelerator_daarem_alpha=float(
            optimizer_kwargs["accelerator_daarem_alpha"]
        ),
        accelerator_daarem_kappa=int(
            optimizer_kwargs["accelerator_daarem_kappa"]
        ),
        accelerator_cycl_monotone_tol=float(
            optimizer_kwargs["accelerator_cycl_monotone_tol"]
        ),
        verbose=verbose,
    )

    # Step 2: Create initial state (this will eventually replace manual initialization)
    torch.set_default_dtype(config.dtype) # TODO: Make this less global
    state = get_initial_state(config)

    # Init
    if config.do_reject:
        raise NotImplementedError(
            "Sample rejection by log likelihood is not yet supported."
        )  # pragma: no cover
    dataseg = X.copy()

    # Whitening
    do_sphere = True if whiten in {"zca", "pca"} else False
    do_approx_sphere = True if whiten == "zca" else False
    do_mean = True if mean_center else False
    dataseg, whitening_matrix, sldet, whitening_inverse, mean = pre_whiten(
        X=dataseg,
        n_components=n_components,
        mineig=mineig,
        do_mean=do_mean,
        do_sphere=do_sphere,
        do_approx_sphere=do_approx_sphere,
        inplace=True,
        )

    # Run AMICA
    state_dict, LL = solve(
        X=dataseg,
        config=config,
        state=state,
        sldet=sldet,
        random_state=random_state,
        initial_weights=w_init,
        initial_scales=sbeta_init,
        initial_locations=mu_init,
        )

    return dict(
        S=whitening_matrix,
        mean=mean,
        gm=state_dict["gm"],
        mu=state_dict["mu"],
        rho=state_dict["rho"],
        sbeta=state_dict["sbeta"],
        W=state_dict["W"],
        A=state_dict["A"],
        c=state_dict["c"],
        alpha=state_dict["alpha"],
        LL=LL,
    )

def solve(
        X,
        *,
        config,
        state,
        sldet,
        random_state=None,
        initial_weights=None,
        initial_scales=None,
        initial_locations=None,
):
    """Run the AMICA algorithm.

    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the features that have to be unmixed. N is the
        number of features, T is the number of samples. X has to be centered
    initial_weights : array-like, shape (n_components, n_components), optional
        Initial weights for the mixture components. If None, weights are initialized
        randomly. This is meant to be used for testing and debugging purposes only.
    initial_scales : array-like, shape (n_components, n_mixtures), optional
        Initial scales (sbeta) for the mixture components. If None, scales are
        initialized randomly. This is meant to be used for testing and debugging
       purposes only.
    initial_locations : array-like, shape (n_components, n_mixtures), optional
        Initial locations (mu) for the mixture components. If None, locations are
        initialized randomly. This is meant to be used for testing and debugging
        purposes only.
    """
    # No-copy (if on CPU)
    X: DataTensor2D = torch.as_tensor(X, dtype=config.dtype, device=config.device)
    rng = torch.Generator(device=torch.device(config.device))
    if random_state is not None:
        rng.manual_seed(random_state)
    # The API will use n_components but under the hood we'll match the Fortran naming
    # TODO: Maybe rename n_components to num_comps in the config dataclass?
    num_comps = config.n_components
    num_mix = config.n_mixtures
    # !-------------------- ALLOCATE VARIABLES ---------------------

    # !------------------- INITIALIZE VARIABLES ----------------------
    # print *, myrank+1, ': Initializing variables ...'; call flush(6);
    state, wc = initialize_state_parameters(
        state=state,
        config=config,
        rng=rng,
        initial_weights=initial_weights,
        initial_scales=initial_scales,
        initial_locations=initial_locations,
    )

    # !-------------------- Determine optimal block size -------------------
    log(f"1: block size = {config.batch_size}", level="info", color=None)

    # !XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX main loop XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    log(
        "Solving. (please be patient, this may take a while)...",
        level="info",
        color="blue",
        weight="bold"
    )
    with torch.no_grad():
        state, LL = optimize(
            X=X,
            sldet=float(sldet),
            wc=wc,
            config=config,
            state=state,
        )
    # Convert Tensors to numpy arrays for output
    state_dict = state.to_numpy()
    LL = LL.cpu().numpy()
    return state_dict, LL


def initialize_state_parameters(
        *,
        state: AmicaState,
        config: AmicaConfig,
        rng: torch.Generator,
        initial_weights=None,
        initial_scales=None,
        initial_locations=None,
) -> tuple[AmicaState, torch.Tensor]:
    """Initialize learnable AMICA parameters and derived unmixing quantities."""
    num_comps = config.n_components
    num_mix = config.n_mixtures
    if not torch.isclose(state.gm.sum(), state.gm.new_tensor(1.0)):
        raise RuntimeError("Initial model weights must sum to 1.0.")

    state.alpha[:, :num_mix] = 1.0 / num_mix
    mu_values = torch.arange(num_mix, dtype=config.dtype, device=config.device)
    mu_values -= (num_mix - 1) / 2
    state.mu[:, :] = mu_values[None, :]

    if initial_locations is None:
        initial_locations = torch.rand(
            num_comps, num_mix, generator=rng, device=config.device, dtype=config.dtype
        )
    else:
        assert initial_locations.shape == (num_comps, num_mix)
        initial_locations = torch.as_tensor(
            initial_locations, dtype=config.dtype, device=config.device
        )
    state.mu = state.mu + 0.05 * (1.0 - 2.0 * initial_locations)

    if initial_scales is None:
        initial_scales = torch.rand(
            num_comps, num_mix, generator=rng, device=config.device, dtype=config.dtype
        )
    else:
        assert initial_scales.shape == (num_comps, num_mix)
        initial_scales = torch.as_tensor(
            initial_scales, dtype=config.dtype, device=config.device
        )
    state.sbeta = 1.0 + 0.1 * (0.5 - initial_scales)

    state.c.fill_(0.0)

    if initial_weights is None:
        initial_weights = torch.rand(
            num_comps,
            num_comps,
            generator=rng,
            device=config.device,
            dtype=config.dtype,
        )
    else:
        assert initial_weights.shape == (num_comps, num_comps)
        initial_weights = torch.as_tensor(
            initial_weights, dtype=config.dtype, device=config.device
        )

    state.A[:, :] = 0.01 * (0.5 - initial_weights)
    idx = torch.arange(num_comps, device=config.device)
    state.A[idx, idx] = 1.0
    Anrmk = torch.linalg.norm(state.A[:, :], dim=0)
    state.A[:, :] /= Anrmk

    W, wc = get_unmixing_matrices(c=state.c, A=state.A, W=state.W)
    assert W.dtype == torch.float64
    state.W = W.clone()
    return state, wc


def optimize(
        *,
        X: DataTensor2D,
        sldet: float,
        wc: torch.Tensor,
        config: AmicaConfig,
        state: AmicaState,
):
    """Optimize the learnable Paramters."""
    # Just set all convergence creterion to the user specific tol
    min_dll = config.tol
    min_nd = config.tol

    # These variables can be updated in the loop
    leave = False
    do_newton = config.do_newton
    numdecs = 0  # number of consecutive iterations where LL decreased from previous
    numincs = 0  # number of consecutive iterations where LL increased by less than tol
    metrics = IterationMetrics(
        iter=1,
        lrate=config.lrate,
        rholrate=config.rholrate,
        lrate0=config.lrate,  # updates slower than lrate..
        rholrate0=config.rholrate,  # Updates slower than rholrate..
        newtrate=config.newtrate,
    )

    # Initialize accumulators container
    accumulators = initialize_accumulators(config)
    device = torch.device(config.device)
    if device.type != "cpu":
        state.to_device(device=device)
        wc = wc.to(device=device)
    # We allocate these separately.
    Dsum = torch.tensor(0.0, dtype=torch.float64, device=config.device)
    Dsign = torch.tensor(1.0, dtype=torch.float64, device=config.device)
    # likelihood history
    LL = torch.zeros(max(1, config.max_iter), dtype=torch.float64, device=config.device)

    c_start = time.time()
    c1 = time.time()
    progress = None
    task_id = None
    if config.verbose == 1:
        progress, task_id = make_progress_bar(
            total=config.max_iter,
            lrate=metrics.lrate,
        )
    try:
        return _main_loop(
            X=X,
            sldet=sldet,
            wc=wc,
            config=config,
            state=state,
            do_newton=do_newton,
            leave=leave,
            numdecs=numdecs,
            numincs=numincs,
            metrics=metrics,
            accumulators=accumulators,
            Dsum=Dsum,
            Dsign=Dsign,
            LL=LL,
            c_start=c_start,
            c1=c1,
            progress=progress,
            task_id=task_id,
            min_dll=min_dll,
            min_nd=min_nd,
        )
    finally:
        if progress is not None:
            progress.stop()


def _main_loop(
        *,
        X: DataTensor2D,
        sldet: float,
        wc: torch.Tensor,
        config: AmicaConfig,
        state: AmicaState,
        do_newton: bool,
        leave: bool,
        numdecs: int,
        numincs: int,
        metrics: IterationMetrics,
        accumulators: AmicaAccumulators,
        Dsum: torch.Tensor,
        Dsign: torch.Tensor,
        LL: torch.Tensor,
        c_start: float,
        c1: float,
        progress,
        task_id,
        min_dll: float,
        min_nd: float,
):
    """Run the AMICA optimization loop and return updated state and LL history."""
    accelerator = None
    if config.optimizer in {"anderson", "daarem"}:
        accelerator = AndersonEMAccelerator(
            order=config.accelerator_order,
            damping=config.accelerator_damping,
            ridge=config.accelerator_ridge,
            monotone=(config.optimizer == "daarem"),
            epsilon_monotone=config.accelerator_eps_monotone,
            restart_on_reject=config.accelerator_history_reset_on_reject,
            max_consecutive_rejects=max(1, config.accelerator_max_restarts),
            daarem_alpha=config.accelerator_daarem_alpha,
            daarem_kappa=config.accelerator_daarem_kappa,
            cycl_monotone_tol=config.accelerator_cycl_monotone_tol,
        )
    while metrics.iter <= config.max_iter:
        previous_state = state.clone()
        step = em_step(
            X=X,
            sldet=sldet,
            wc=wc,
            config=config,
            state=state,
            iteration=metrics.iter,
            do_newton=do_newton,
            accumulators=accumulators,
            lrate=metrics.lrate,
            rholrate=metrics.rholrate,
            lrate0=metrics.lrate0,
            rholrate0=metrics.rholrate0,
            newtrate=metrics.newtrate,
        )
        state = step.state
        wc = step.wc
        metrics.loglik = step.likelihood
        metrics.ndtmpsum = step.ndtmpsum
        metrics.lrate = step.lrate
        metrics.rholrate = step.rholrate

        ndtmpsum = metrics.ndtmpsum
        LL[metrics.iter - 1] = metrics.loglik

        # !----- display log likelihood of data
        # if (seg_rank == 0) then
        c2 = time.time()
        t0 = c2 - c1
        #  if (mod(iter,outstep) == 0) then

        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                completed=metrics.iter,
                ll=f"{float(LL[metrics.iter - 1]):.4f}",
                nd=f"{float(ndtmpsum):.4f}",
                lrate=f"{metrics.lrate:.5f}",
            )

        if config.verbose == 2 and (metrics.iter % outstep) == 0:
            report = (
                f"Iteration {metrics.iter}, "
                f"lrate = {metrics.lrate:.5f}, "
                f"LL = {LL[metrics.iter - 1]:.7f}, "
                f"nd = {ndtmpsum:.7f}, D = {float(Dsum):.5f} "
                f"took {t0:.2f} seconds"
            )
            log(msg=report, level="info", color=None)
            c1 = time.time()

        # !----- check whether likelihood is increasing
        # if (seg_rank == 0) then
        # ! if we get a NaN early, try to reinitialize and startover a few times
        if torch.isnan(LL[metrics.iter - 1]):
            raise RuntimeError(f"Log Likelihood is NaN at iteration {metrics.iter}")
        # end if
        if metrics.iter > 1:
            if (LL[metrics.iter - 1] < LL[metrics.iter - 2]):
                log("Likelihood decreasing!", level="warning", color="yellow")
                if (metrics.lrate < minlrate) or (ndtmpsum <= min_nd):
                    leave = True
                    log(
                        "minimum change threshold met, exiting loop",
                        level="info",
                        color="green",
                        weight="bold"
                        )
                else:
                    metrics.lrate *= lratefact
                    metrics.rholrate *= rholratefact
                    numdecs += 1
                    if numdecs >= maxdecs:
                        metrics.lrate0 *= lratefact
                        if metrics.iter > config.newt_start:
                            metrics.rholrate0 *= rholratefact
                        if config.do_newton and metrics.iter > config.newt_start:
                            log(
                                "Reducing maximum Newton lrate",
                                level="info",
                                color="blue"
                                )
                            metrics.newtrate *= lratefact
                        numdecs = 0
                    # end if (numdecs >= maxdecs)
                # end if (lrate vs minlrate)
            # end if LL
            if use_min_dll:
                if (LL[metrics.iter - 1] - LL[metrics.iter - 2]) < min_dll:
                    numincs += 1
                    if numincs > maxincs:
                        leave = True
                        log(
                            "Exiting because likelihood increasing by less than "
                            f"{min_dll} for more than {maxincs} iterations ...",
                            level="info",
                            color="green",
                            weight="bold"
                            )
                else:
                    numincs = 0
            else:
                raise NotImplementedError()  # pragma: no cover
            if use_grad_norm:
                if ndtmpsum < min_nd:
                    leave = True
                    log(
                        "Exiting because norm of weight gradient less than "
                        f"{min_nd:.12f}",
                        level="info",
                        color="green",
                        weight="bold",
                        )
        # end if (iter > 1)
        if config.do_newton and (metrics.iter == config.newt_start):
            log("Starting Newton ... setting numdecs to 0", level="info", color="blue")
            numdecs = 0
        # call MPI_BCAST(leave,1,MPI_LOGICAL,0,seg_comm,ierr)
        # call MPI_BCAST(startover,1,MPI_LOGICAL,0,seg_comm,ierr)
        if leave:
            c_end = time.time()
            _emit_status(progress, f"Finished in {c_end - c_start:.2f} seconds")
            return state, LL

        state, wc, accel_outcome = maybe_apply_acceleration(
            accelerator=accelerator,
            config=config,
            X=X,
            sldet=sldet,
            previous_state=previous_state,
            current_state=state,
            current_loglik=float(metrics.loglik),
            iteration=metrics.iter,
        )

        # !----- reject data
        if config.do_reject:
            raise NotImplementedError()  # pragma: no cover

        if config.verbose == 2 and accel_outcome.attempted:
            parts = [
                f"iter {metrics.iter}",
                f"EM ll={float(metrics.loglik):.7f}",
                f"accel={config.optimizer}",
                f"hist={accel_outcome.history}",
                f"accepted={accel_outcome.accepted}",
            ]
            if accel_outcome.candidate_loglik is not None:
                parts.append(f"cand_ll={accel_outcome.candidate_loglik:.7f}")
            parts.append(f"reason={accel_outcome.reason}")
            if accel_outcome.restart:
                parts.append("restart=True")
            log(" | ".join(parts), level="info", color=None)

        metrics.iter += 1
        # end if/else
    # end while
    warn(
        "Maximum number of iterations reached before convergence."
        " Consider increasing max_iter or relaxing tol.",
        ConvergenceWarning,
    )
    c_end = time.time()
    _emit_status(progress, f"Finished in {c_end - c_start:.2f} seconds")
    return state, LL


def em_step(
        *,
        X: DataTensor2D,
        sldet: float,
        wc: torch.Tensor,
        config: AmicaConfig,
        state: AmicaState,
        iteration: int,
        do_newton: bool,
        accumulators: AmicaAccumulators,
        lrate: float,
        rholrate: float,
        lrate0: float,
        rholrate0: float,
        newtrate: float,
) -> EMStepResult:
    """Run one full AMICA outer EM iteration as a side-effect-light map."""
    state = state.clone()
    accumulators.reset()
    total_LL = torch.zeros((), dtype=config.dtype, device=config.device)
    doing_newton = do_newton and (iteration >= config.newt_start)
    _, Dsum = compute_sign_log_determinant(unmixing_matrix=state.W, minlog=minlog)

    initial = get_initial_model_log_likelihood(
        unmixing_logdet=Dsum,
        whitening_logdet=sldet,
        model_weight=state.gm[0],
    )

    batch_loader = BatchLoader(X, axis=0, batch_size=config.batch_size)
    for data_batch, _ in batch_loader:
        if state.W.device.type != data_batch.device.type:
            raise ValueError(
                f"Mismatch between state.W device ({state.W.device}) "
                f"and data_batch device ({data_batch.device})"
            )
        b = compute_preactivations(
            X=data_batch,
            unmixing_matrix=state.W,
            bias=wc,
            do_reject=config.do_reject,
            n_weights=config.n_components,
        )
        scratch_shape = (
            data_batch.shape[0],
            config.n_components,
            config.n_mixtures,
        )
        _ensure_batch_scratch(
            accumulators,
            shape=scratch_shape,
            dtype=config.dtype,
            device=config.device,
        )
        y, z = compute_source_densities(
            pdftype=config.pdftype,
            b=b,
            sbeta=state.sbeta,
            mu=state.mu,
            alpha=state.alpha,
            rho=state.rho,
            out_sources=accumulators.scratch_y,
            out_logits=accumulators.scratch_z,
        )
        likelihood_sum, z = compute_loglikelihood_and_responsibilities(
            log_densities=z,
            initial_loglik=initial,
        )
        total_LL += likelihood_sum
        vsum = torch.as_tensor(
            data_batch.shape[0],
            dtype=config.dtype,
            device=config.device,
        )
        u = compute_weighted_responsibilities(
            mixture_responsibilities=z,
            single_model=True,
        )
        usum = u.sum(dim=0)

        fp = compute_source_scores(
            pdftype=config.pdftype,
            y=y,
            rho=state.rho,
            out_scores=accumulators.scratch_fp,
        )
        ufp = precompute_weighted_scores(
            weighted_responsibilities=u,
            scores=fp,
            out_ufp=fp if not doing_newton else accumulators.scratch_ufp,
        )
        fp_for_mu = fp if doing_newton else ufp

        g = compute_scaled_scores(weighted_scores=ufp, scales=state.sbeta)
        accumulators.dgm_numer[0] += vsum
        accumulate_c_stats(
            X=data_batch,
            vsum=vsum,
            n_weights=config.n_components,
            out_numer=accumulators.dc_numer,
            out_denom=accumulators.dc_denom,
        )
        accumulate_alpha_stats(
            usum=usum,
            vsum=vsum,
            out_numer=accumulators.dalpha_numer,
            out_denom=accumulators.dalpha_denom,
        )
        accumulate_mu_stats(
            ufp=ufp,
            rho=state.rho,
            sbeta=state.sbeta,
            y=y,
            fp=fp_for_mu,
            out_numer=accumulators.dmu_numer,
            out_denom=accumulators.dmu_denom,
        )
        accumulate_beta_stats(
            usum=usum,
            rho=state.rho,
            u=u,
            ufp=ufp,
            y=y,
            out_numer=accumulators.dbeta_numer,
            out_denom=accumulators.dbeta_denom,
        )
        accumulate_rho_stats(
            y=y,
            rho=state.rho,
            u=u,
            usum=usum,
            epsdble=epsdble,
            out_numer=accumulators.drho_numer,
            out_denom=accumulators.drho_denom,
        )
        if doing_newton:
            accumulate_sigma2_stats(
                source_estimates=b,
                vsum=vsum,
                out_numer=accumulators.newton.dsigma2_numer,
                out_denom=accumulators.newton.dsigma2_denom,
            )
            accumulate_kappa_stats(
                ufp=ufp,
                fp=fp,
                sbeta=state.sbeta,
                usum=usum,
                out_numer=accumulators.newton.dkappa_numer,
                out_denom=accumulators.newton.dkappa_denom,
            )
            accumulate_lambda_stats(
                fp=fp,
                y=y,
                u=u,
                usum=usum,
                out_numer=accumulators.newton.dlambda_numer,
                out_denom=accumulators.newton.dlambda_denom,
            )
            accumulators.newton.dbaralpha_numer[:, :] += usum
            accumulators.newton.dbaralpha_denom[:, :] += vsum
        else:
            fp = None
        accumulators.dA[:, :] += torch.matmul(g.T, b)

    likelihood, ndtmpsum = accum_updates_and_likelihood(
        X=X,
        config=config,
        accumulators=accumulators,
        state=state,
        total_LL=total_LL,
        iteration=iteration,
    )
    lrate, rholrate, state, wc = update_params(
        X=X,
        iteration=iteration,
        config=config,
        state=state,
        accumulators=accumulators,
        lrate=lrate,
        rholrate=rholrate,
        lrate0=lrate0,
        rholrate0=rholrate0,
        wc=wc.clone(),
        newtrate=newtrate,
    )
    return EMStepResult(
        state=state,
        wc=wc,
        likelihood=likelihood,
        ndtmpsum=ndtmpsum,
        lrate=lrate,
        rholrate=rholrate,
    )


def evaluate_loglikelihood(
        *,
        X: DataTensor2D,
        sldet: float,
        config: AmicaConfig,
        state: AmicaState,
) -> torch.Tensor:
    """Compute observed-data log-likelihood for a fixed AMICA parameter state."""
    _, Dsum = compute_sign_log_determinant(unmixing_matrix=state.W, minlog=minlog)
    _, wc = get_unmixing_matrices(c=state.c, A=state.A, W=state.W)
    initial = get_initial_model_log_likelihood(
        unmixing_logdet=Dsum,
        whitening_logdet=sldet,
        model_weight=state.gm[0],
    )
    total = torch.tensor(0.0, dtype=config.dtype, device=config.device)
    batch_loader = BatchLoader(X, axis=0, batch_size=config.batch_size)
    for data_batch, _ in batch_loader:
        b = compute_preactivations(
            X=data_batch,
            unmixing_matrix=state.W,
            bias=wc,
            do_reject=config.do_reject,
            n_weights=config.n_components,
        )
        _, z = compute_source_densities(
            pdftype=config.pdftype,
            b=b,
            sbeta=state.sbeta,
            mu=state.mu,
            alpha=state.alpha,
            rho=state.rho,
        )
        total += compute_model_loglikelihood_sum(
            log_densities=z,
            initial_loglik=initial,
        )
    return total / (X.shape[0] * config.n_components)


def maybe_apply_acceleration(
        *,
        accelerator: AndersonEMAccelerator | None,
        config: AmicaConfig,
        X: DataTensor2D,
        sldet: float,
        previous_state: AmicaState,
        current_state: AmicaState,
        current_loglik: float,
        iteration: int,
) -> tuple[AmicaState, torch.Tensor, AccelerationOutcome]:
    """Apply optional post-EM Anderson / DAAREM-style extrapolation."""
    _, wc = get_unmixing_matrices(
        c=current_state.c,
        A=current_state.A,
        W=current_state.W,
    )
    if accelerator is None:
        return current_state, wc, AccelerationOutcome(reason="disabled")
    x_prev = pack_state(previous_state)
    g_curr = pack_state(current_state)
    accelerator.update(x=x_prev, g=g_curr)
    if iteration < config.accelerator_start_iter:
        return current_state, wc, AccelerationOutcome(reason="warmup")
    if (
        (iteration - config.accelerator_start_iter)
        % max(1, config.accelerator_period)
        != 0
    ):
        return current_state, wc, AccelerationOutcome(reason="period")

    outcome = AccelerationOutcome(attempted=True, reason="insufficient_history")
    proposal = accelerator.propose()
    if proposal is None:
        return current_state, wc, outcome
    outcome.history = proposal.history

    candidate_state = unpack_state(proposal.candidate, current_state)
    valid, reason = is_valid_state(candidate_state)
    if not valid:
        outcome.reason = reason
        outcome.restart = accelerator.reject()
        return current_state, wc, outcome

    if config.accelerator_validate_candidate:
        candidate_loglik = float(
            evaluate_loglikelihood(
                X=X,
                sldet=sldet,
                config=config,
                state=candidate_state,
            )
        )
        outcome.candidate_loglik = candidate_loglik
        if config.optimizer == "daarem":
            if candidate_loglik < current_loglik - config.accelerator_eps_monotone:
                outcome.reason = "monotonicity"
                accelerator.update_cycle_monotonicity(
                    loglik=current_loglik,
                    history=proposal.history,
                )
                outcome.restart = accelerator.reject()
                return current_state, wc, outcome
        outcome.reason = "validated"
    else:
        outcome.reason = "sanity_only"

    candidate_w, candidate_wc = get_unmixing_matrices(
        c=candidate_state.c, A=candidate_state.A, W=candidate_state.W
    )
    candidate_state.W = candidate_w
    accelerator.accept()
    if (
        config.optimizer == "daarem"
        and config.accelerator_validate_candidate
        and outcome.candidate_loglik is not None
    ):
        accelerator.update_cycle_monotonicity(
            loglik=outcome.candidate_loglik,
            history=proposal.history,
        )
    outcome.accepted = True
    return candidate_state, candidate_wc, outcome


def accum_updates_and_likelihood(
        *,
        X,
        config,
        accumulators,
        state,
        total_LL,  # this is LLtmp in Fortran
        iteration
        ):
    """Use accumulated arrays to updated logk and ndtmpsum."""
    # !--- add to the cumulative dtmps
    # ...
    #--------------------------FORTRAN CODE-------------------------
    # call MPI_REDUCE(dgm_numer_tmp,dgm_numer,num_models,MPI_DOUBLE_PRECISION,MPI_S...
    # ...
    # if update_A:
    # call MPI_REDUCE(dWtmp,dA,nw*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_co...
    nw = config.n_components
    Wtmp_working = torch.zeros(
        (config.n_components, config.n_components),
        dtype=config.dtype, device=config.device
        )
    # if (seg_rank == 0) then
    if config.do_newton and iteration >= config.newt_start:
        newton_terms = compute_newton_terms(
            accumulators=accumulators, config=config, mu=state.mu
            )

        sigma2 = newton_terms["sigma2"]
        kappa = newton_terms["kappa"]
        lambda_ = newton_terms["lambda_"]
        # if (print_debug) then
    # end if (do_newton .and. iter >= newt_start)

    #--------------------------FORTRAN CODE-------------------------
    # if (print_debug) then
    # print *, 'dA ', h, ' = '; call flush(6)
    # call DSCAL(nw*nw,dble(-1.0)/dgm_numer(h),dA(:,:,h),1)
    # dA(i,i,h) = dA(i,i,h) + dble(1.0)
    #---------------------------------------------------------------
    if config.do_reject:
        raise NotImplementedError()  # pragma: no cover
    else:
        accumulators.dA[:, :] *= -1.0 / accumulators.dgm_numer[0]

    # basically the same as np.fill_diagonal where fill value is diag + 1.0
    diag = accumulators.dA.diagonal()
    idx = torch.arange(nw)
    accumulators.dA[idx, idx] = diag + 1.0
    # if (print_debug) then

    if config.do_newton and iteration >= config.newt_start:
        #--------------------------FORTRAN CODE-------------------------
        # do i = 1,nw ... do k = 1,nw
        # if (i == k) then
        # Wtmp(i,i) = dA(i,i,h) / lambda(i,h)
        # else
        # sk1 = sigma2(i,h) * kappa(k,h)
        # sk2 = sigma2(k,h) * kappa(i,h)
        #---------------------------------------------------------------
        # on-diagonal elements
        diag = accumulators.dA.diagonal()
        fill_values = diag / lambda_
        idx = torch.arange(Wtmp_working.shape[0])
        Wtmp_working[idx, idx] = fill_values

        # off-diagonal elements
        i_indices, k_indices = torch.meshgrid(
            torch.arange(config.n_components, device=config.device),
            torch.arange(config.n_components, device=config.device), indexing='ij',
            )
        off_diag_mask = i_indices != k_indices
        sk1 = sigma2[i_indices] * kappa[k_indices]
        sk2 = sigma2[k_indices] * kappa[i_indices]
        positive_mask = (sk1 * sk2 > 0.0)
        if torch.any(~positive_mask):
            raise RuntimeError(
                "Non-positive definite Hessian encountered in Newton update. "
                f"Iteration {iteration}. Try setting do_newton to False."
                )
        condition_mask = positive_mask & off_diag_mask
        if torch.any(condition_mask):
            # # Wtmp(i,k) = (sk1*dA(i,k,h) - dA(k,i,h)) / (sk1*sk2 - dble(1.0))
            numerator = (
                sk1
                * accumulators.dA[i_indices, k_indices]
                - accumulators.dA[k_indices, i_indices]
                )
            denominator = sk1 * sk2 - 1.0
            Wtmp_working[condition_mask] = (numerator / denominator)[condition_mask]
        # end if (i == k)
        # end do (k)
        # end do (i)
    # end if (do_newton .and. iter >= newt_start)
    if ((not config.do_newton) or (iteration < config.newt_start)):
        #  Wtmp = dA(:,:,h)
        assert Wtmp_working.shape == accumulators.dA.shape == (nw, nw)
        Wtmp_working = accumulators.dA.clone()
        assert Wtmp_working.shape == (nw, nw)
    #--------------------------FORTRAN CODE-------------------------
    # call DSCAL(nw*nw,dble(0.0),dA(:,:,h),1)
    # call DGEMM('N','N',nw,nw,nw,dble(1.0),A(:,comp_list(:,h)),nw,Wtmp,nw,dble...
    #---------------------------------------------------------------
    accumulators.dA[:, :] = 0.0
    accumulators.dA[:, :] += torch.matmul(state.A, Wtmp_working)

    zeta = torch.zeros(config.n_components, dtype=config.dtype, device=config.device)
    #--------------------------FORTRAN CODE-------------------------
    # dAk(:,comp_list(i,h)) = dAk(:,comp_list(i,h)) + gm(h)*dA(:,i,h)
    # zeta(comp_list(i,h)) = zeta(comp_list(i,h)) + gm(h)
    #---------------------------------------------------------------
    source_columns = state.gm[0] * accumulators.dA
    accumulators.dAK[:, :] += source_columns
    zeta[:] += state.gm[0]

    #--------------------------FORTRAN CODE-------------------------
    # dAk(:,k) = dAk(:,k) / zeta(k)
    # nd(iter,:) = sum(dAk*dAk,1)
    # ndtmpsum = sqrt(sum(nd(iter,:),mask=comp_used) / (nw*count(comp_used)))
    #---------------------------------------------------------------
    accumulators.dAK[:,:] /= zeta  # Broadcasting division
    # nd is (num_iters, num_comps) in Fortran, but we only store current iteration
    nd = torch.sum(accumulators.dAK * accumulators.dAK, dim=0)
    assert nd.shape == (config.n_components,)

    # comp_used should be a vector of True
    # In Fortran comp_used was based on component availability.
    # Unless identify_shared_comps was run. I have no plans to implement that.
    comp_used = torch.ones(config.n_components, dtype=bool)
    assert isinstance(comp_used, torch.Tensor)
    assert comp_used.shape == (config.n_components,)
    assert comp_used.dtype == torch.bool
    ndtmpsum = torch.sqrt(torch.sum(nd) / (nw * torch.count_nonzero(comp_used)))
    # end if (update_A)

    # if (seg_rank == 0) then
    if config.do_reject:
        raise NotImplementedError()  # pragma: no cover
    else:
        # LL(iter) = LLtmp2 / dble(all_blks*nw)
        # XXX: In the Fortran code LLtmp2 is the summed LLtmps across processes.
        likelihood = total_LL / (X.shape[0] * nw)
    return (likelihood, ndtmpsum)


def update_params(
        *,
        X,
        iteration,
        config,
        state,
        accumulators,
        lrate,
        rholrate,
        lrate0,
        rholrate0,
        newtrate,
        wc,
):
    """Update learnable ICA Parameters, and learning rates."""
    # if (seg_rank == 0) then
    # if update_gm:
    if config.do_reject:
        raise NotImplementedError()  # pragma: no cover
        # gm = dgm_numer / dble(numgoodsum)
    else:
        state.gm[:] = accumulators.dgm_numer / X.shape[0]
    # end if (update_gm)

    # if update_alpha:
    # assert alpha.shape == (num_comps, num_mix)
    state.alpha[:, :] = accumulators.dalpha_numer / accumulators.dalpha_denom
    if torch.any(~torch.isfinite(state.alpha)):
        raise RuntimeError("Non-finite alpha encountered during update.")

    # if update_c:
    # assert c.shape == (nw, num_models)
    state.c[:] = accumulators.dc_numer / accumulators.dc_denom
    if torch.any(~torch.isfinite(state.c)):
        raise RuntimeError("Non-finite c encountered during update.")

    # === Section: Apply Parameter accumulators & Rescale ===
    # Apply accumulated statistics to update parameters, then rescale and refresh W/wc.
    # !print *, 'updating A ...'; call flush(6)
    if (iteration < share_start or (iteration % share_iter > 5)):
        if config.do_newton and (iteration >= config.newt_start):
            # lrate = min( newtrate, lrate + min(dble(1.0)/dble(newt_ramp),lrate) )
            # rholrate = rholrate0
            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            lrate = min(newtrate, lrate + min(1.0 / config.newt_ramp, lrate))
            rholrate = rholrate0
            state.A -= lrate * accumulators.dAK
        else:
            lrate = min(lrate0, lrate + min(1 / config.newt_ramp, lrate))
            rholrate = rholrate0
            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            state.A -= lrate * accumulators.dAK
        # end if do_newton
    # end if (update_A)

    # if update_mu:
    state.mu += accumulators.dmu_numer / accumulators.dmu_denom
    if torch.any(~torch.isfinite(state.mu)):
        raise RuntimeError("Non-finite mu encountered during update.")

    # if update_beta:
    state.sbeta *= torch.sqrt(accumulators.dbeta_numer / accumulators.dbeta_denom)
    sbetatmp = torch.minimum(state.sbeta.new_tensor(invsigmax), state.sbeta)
    state.sbeta = torch.maximum(state.sbeta.new_tensor(invsigmin), sbetatmp)
    if torch.any(~torch.isfinite(state.sbeta)):
        raise RuntimeError("Non-finite sbeta encountered during update.")


    state.rho += (
            rholrate
            * (
                1.0
                - (state.rho / torch.special.psi(1.0 + 1.0 / state.rho))
            * accumulators.drho_numer
            / accumulators.drho_denom
        )
    )
    rhotmp = torch.minimum(state.rho.new_tensor(maxrho), state.rho) # shape (num_comps, num_mix)
    assert rhotmp.shape == (config.n_components, config.n_mixtures)
    state.rho = torch.maximum(state.rho.new_tensor(minrho), rhotmp)

    # !--- rescale
    # !print *, 'rescaling A ...'; call flush(6)
    # from seed import A_FORTRAN
    if doscaling:
        # calculate the L2 norm for each column of A and then use it to normalize that
        # column and scale the corresponding columns in mu and sbeta, but only if the
        # norm is positive.
        Anrmk = torch.linalg.norm(state.A, dim=0)
        positive_mask = Anrmk > 0
        if positive_mask.all():
            state.A[:, positive_mask] /= Anrmk[positive_mask]
            state.mu[positive_mask, :] *= Anrmk[positive_mask, None]
            state.sbeta[positive_mask, :] /= Anrmk[positive_mask, None]
        else:
            raise NotImplementedError()  # pragma: no cover
    # end if (doscaling)

    if share_comps:
        raise NotImplementedError()  # pragma: no cover

    state.W, wc = get_unmixing_matrices(
        c=state.c,
        A=state.A,
        W=state.W,
    )
    # if (print_debug) then
    # call MPI_BCAST(gm,num_models,MPI_DOUBLE_PRECISION,0,seg_comm,ierr)
    # ...
    return lrate, rholrate, state, wc
