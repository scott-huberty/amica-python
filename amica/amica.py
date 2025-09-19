from copy import copy
from pathlib import Path
import time
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

import torch

from _batching import BatchLoader, choose_batch_size
from constants import (
    fix_init,
    mineig,
    minlog,
    epsdble,
    maxrej,
    rejstart,
    rejint,
    doscaling,
    share_comps,
    share_start,
    share_iter,
    minrho,
    maxrho,
    invsigmin,
    invsigmax,
    use_min_dll,
    use_grad_norm,
    maxincs,
    maxdecs,
    outstep,
    restartiter,
    numrestarts,
    maxrestarts,
    minlrate,
    min_nd,
    lratefact,
    rholratefact,
    LOG_2,
    LOG_SQRT_PI,
)

from seed import MUTMP, SBETATMP as sbetatmp, WTMP

from state import (
    AmicaConfig,
    AmicaState,
    IterationMetrics,
    get_initial_state,
    initialize_accumulators,
)
from _typing import (
    SamplesVector,
    ComponentsVector,
    DataArray2D,
    DataTensor2D,
    WeightsArray,
    SourceArray2D,
    SourceArray3D,
    ParamsArray,
    ParamsModelArray,
    ParamsModelTensor,
    LikelihoodArray,
    ScalarTensor,
)

import line_profiler

sbetatmp = sbetatmp.T

import warnings
warnings.filterwarnings("error")


def get_component_slice(h: int, n_components: int) -> slice:
    """Return slice for components of model h.

    Parameters
    - h: model number (1-based)
    - n_components: number of components per model (nw)

    Returns
    - slice object for components of model h

    Notes
    -----
    - Creating a slice ensures that we get a view when indexing arrays.
    - The fortran code pre-computes comp_list(num_components, num_models). We avoid this
        by computing the slice on-the-fly and thus avoiding fancy indexing.
    
    Fortran reference:
        do h = 1,num_models
            do i = 1,nw
                comp_list(i,h) = (h-1) * nw + i
    """
    h_index = h - 1  # Convert to 0-based index
    start = h_index * n_components
    end = start + n_components
    return slice(start, end)


# TODO's
# - consider keeping z and z0 separate once chunking is implemented
# - consider giving g its own array one chunking is implemented
# - if we never chunk process the M-step, then numer/denom accumulators are once per iteration
#   so we can assign directly instead of accumulating e.g. dalpha_denom.fill(vsum)

def amica(
        X,
        *,
        whiten=True,
        centering=True,
        n_components=None,
        n_models=1,
        n_mixtures=3,
        max_iter=500,
        tol=1e-7,
        lrate=0.05,
        rholrate=0.05,
        pdftype=0,
        do_newton=True,
        newt_start=50,
        newtrate=1,
        newt_ramp=10,
        batch_size=None,
        do_reject=False,
        random_state=None,
):
    """Perform Adaptive Mixture Independent Component Analysis (AMICA).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and
        n_features is the number of features.
    n_components : int, optional
        Number of components to extract. If None, all components are used.
    n_mixtures: int, optional
        Number of mixtures to use in the AMICA algorithm.
    batch_size : int, optional
        Chunk size for processing data in chunks along the samples axis. If ``None``,
        chunking is chosen automatically to keep peak memory under ~1.5 GB, and
        warns if the chunk size is below ~8k samples (If the input data is small enough
        to process in one shot, no chunking is used). If you want to enforce no chunking,
        you can override this memory cap by setting batch_size explicitly, e.g. to
        `X.shape[1]` to process all samples at once.", but note that this may lead to
        high memory usage for large datasets.
    whiten : boolean, optional
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white,
        otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.
    centering : bool, optional
        If True, X is mean corrected.
    n_mixtures : int, optional
        Number of mixtures to use in the AMICA algorithm.
        Default is 1.
    max_iter : int, optional
        Maximum number of iterations to perform. Default is 500.
    random_state : int, RandomState instance or None, optional (default=None)
        Used to perform a random initialization when w_init is not provided.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    if batch_size is None:
        batch_size = choose_batch_size(
            N=X.shape[1],
            n_comps=n_components if n_components is not None else X.shape[0],
            n_mix=n_mixtures,
        )
    # Step 1: Create config and state objects (new dataclass approach)
    config = AmicaConfig(
        n_features=X.shape[0],  # Number of channels (corrected from X.shape[1])
        n_components=n_components if n_components is not None else X.shape[0],
        n_models=n_models,
        n_mixtures=n_mixtures,
        max_iter=max_iter,
        batch_size=batch_size,
        pdftype=pdftype,
        tol=tol,
        lrate=lrate,
        rholrate=rholrate,
        do_newton=do_newton,
        newt_start=newt_start,
        newtrate=newtrate,
        newt_ramp=newt_ramp,
        do_reject=do_reject,
    )
    
    # Step 2: Create initial state (this will eventually replace manual initialization)
    torch.set_default_dtype(config.dtype)
    state = get_initial_state(config)
    
    # random_state = check_random_state(random_state)

    # Init
    if n_models > 1:
        raise NotImplementedError("n_models > 1 not yet supported")
    if do_reject:
        raise NotImplementedError("Sample rejection by log likelihood is not yet supported yet")
    dataseg = X.copy()

    do_mean = True if centering else False
    do_sphere = True if whiten else False
    dataseg, whitening_matrix, sldet, whitening_inverse, mean = pre_whiten(
        X=dataseg,
        n_components=n_components,
        mineig=mineig,
        do_mean=do_mean,
        do_sphere=do_sphere,
        do_approx_sphere=False,
        inplace=True,
        )

    state_dict, LL = _core_amica(
        X=dataseg,
        config=config,
        state=state,
        sldet=sldet,
        )
    gm = state_dict["gm"]
    mu = state_dict["mu"]
    rho = state_dict["rho"]
    sbeta = state_dict["sbeta"]
    W = state_dict["W"]
    A = state_dict["A"]
    c = state_dict["c"]
    alpha = state_dict["alpha"]
    return whitening_matrix, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL


def pre_whiten(
        *,
        X: DataArray2D,
        n_components: Optional[int] = None,
        mineig: float = 1e-6,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_approx_sphere: bool = False,
        inplace: bool = True,
) -> Tuple[DataArray2D, WeightsArray, float, WeightsArray, ComponentsVector | None]:
    """
    Pre-whiten the input data matrix X prior to ICA.
    
    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        Input data matrix to be whitened. If inplace is True, X will be mutated and
        returned as the whitened data. Otherwise a copy will be made and returned.
    n_components : int or None
        Number of components to keep. If None, all components are kept.
    mineig : float
        Minimum eigenvalue threshold for keeping components. Eigenvalues below this will
        be discarded.
    do_mean : bool
        If True, mean-center the data before whitening.
    do_sphere : bool
        If True, perform sphering (whitening). If False, only variance normalization
        is performed (not implemented)
    do_approx_sphere : bool
        If True, use approximate sphering (not implemented).
    inplace : bool
        If True, modify X in place. If False, make a copy of X and modify that.
    
    Returns
    -------
    X : array, shape (n_features, n_samples)
        The whitened data matrix. This is a copy of the input data if inplace is False,
        otherwise it is the mutated input data itself.
    whitening_matrix : array, shape (n_features, n_features)
        The whitening/sphering matrix applied to the data. If do_sphere is False, then
        this is the variance normalization matrix (not implemented).
    sldet : float
        The log-determinant of the whitening matrix.
    whitening_inverse : array, shape (n_features, n_features)
        The pseudoinverse of the whitening matrix. Only returned if do_sphere is True.
        otherwise None.
    mean : array, shape (n_features,)
        The mean of each feature that was subtracted if do_mean is True. Only returned
        if do_mean is True, otherwise None.
    """
    dataseg = X if inplace else X.copy()
    # !---------------------------- get the mean --------------------------------
    nx, n_samples = dataseg.shape
    if n_components is None:
        n_components = nx
    
    # ---- Mean-centering ----
    if do_mean:
        print("getting the mean ...")
        mean = dataseg.mean(axis=1)
        # !--- subtract the mean
        dataseg -= mean[:, None]  # Subtract mean from each channel

    # ---- Covariance ----
    print(" Getting the covariance matrix ...")
    # Compute the covariance matrix
    # The Fortran code only accumulators the lower triangular part of the covariance matrix

    # -------------------- FORTRAN CODE ---------------------------------------
    # call DSCAL(nx*nx,dble(0.0),Stmp,1)
    # call DSYRK('L','N',nx,blk_size(seg),dble(1.0),dataseg(seg)%data(:,bstrt:bstp)...
    # call DSCAL(nx*nx,dble(1.0)/dble(cnt),S,1)
    #------------------------------------------------------------------------
    Cov = dataseg @ dataseg.T / n_samples

    # ---- Eigen-decomposition
    print(f"doing eig nx = {nx}")
    eigvals, eigvecs = np.linalg.eigh(Cov) # ascending order

    min_eigs = eigvals[:min(nx//2, 3)]
    max_eigs = eigvals[::-1][:3]
    print(f"minimum eigenvalues: {min_eigs}")
    print(f"maximum eigenvalues: {max_eigs}")

    min_eigs_fortran = [4.8799005132501803, 6.9201197127079803, 7.6562147928880702]
    max_eigs_fortran = [9711.1430838537090, 3039.6850435125002, 1244.4129447052057]

    
    # keep only valid eigs  
    numeigs = min(n_components, sum(eigvals > mineig)) # np.linalg.matrix_rank?
    print(f"num eigvals kept: {numeigs}")

    # Log determinant of the whitening matrix
    sldet = -0.5 * np.sum(np.log(eigvals))

    # ---- Sphere or variance normalize ----
    if do_sphere:
        print("Sphering the data...")
        if numeigs == nx:
            # call DSCAL(nx*nx,dble(0.0),S,1)
            if do_approx_sphere:
                raise NotImplementedError()
            else:
                # call DCOPY(nx*nx,Stmp2,1,S,1)
                whitening_matrix = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
        else:
            # if (do_approx_sphere) then
            raise NotImplementedError()
    else:
        # !--- just normalize by the channel variances (don't sphere)
        raise NotImplementedError()

    # -------------------- FORTRAN CODE ---------------------------------------
    # call DSCAL(nx*blk_size(seg),dble(0.0),xtmp(:,1:blk_size(seg)),1)
    # call DGEMM('N','N',nx,blk_size(seg),nx,dble(1.0),S,nx,dataseg(seg)%data(:,bstrt:bstp),nx,dble(1.0),xtmp(:,1:blk_size(seg)),nx)
    # call DCOPY(nx*blk_size(seg),xtmp(:,1:blk_size(seg)),1,dataseg(seg)%data(:,bstrt:bstp),1)
    # -------------------------------------------------------------------------
    dataseg = whitening_matrix @ dataseg # Apply the sphering matrix

    # Lets check dataseg


    nw = numeigs # Number of weights, as per Fortran code
    print(f"numeigs = {numeigs}, nw = {nw}")

    # ! get the pseudoinverse of the sphering matrix
    # call DGESVD( 'A', 'S', numeigs, nx, Stmp2, nx, eigvals, sUtmp, numeigs, sVtmp, numeigs, work, lwork, info )
    Winv = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T  # Inverse of the whitening matrix 

    if n_components is None:
        # TODO
        # In the Fortran code n_components == nw IF num_models == 1
        # if num_models > 1, num_comps is nw*num_models
        # Arrays like A(nw, num_comps), and alpha/mu/sbeta/rho(:, num_comps) use this.
        # Indexing: comp_list(i,h) = (h-1)*nw + i ranges up to nw*num_models.
        # So for num_model > 1, we should refactor those arrays to have a num_models dim.
        n_components = nw
    elif n_components > nw:
        raise ValueError(f"n_components must be less than or equal to the rank of the data: {nw}")
    
    if not do_mean:
        mean = None
    return dataseg, whitening_matrix, sldet, Winv, mean


def _core_amica(
        X,
        *,
        config,
        state,
        sldet,
):
    """Runs the AMICA algorithm.
    
    Parameters
    ----------
    X : array, shape (N, T)
        Matrix containing the features that have to be unmixed. N is the
        number of features, T is the number of samples. X has to be centered
    num_comps : int or None
        Number of components to use. If None, it is set to the number of
        features (N) times the number of models.
    max_iter : int
        Maximal number of iterations for the algorithm
    tol : float
        Tolerance for convergence. Iterations stop when the change in log-likelihood
        is less than tol.
    """
    X: DataTensor2D = torch.as_tensor(X, dtype=config.dtype)
    # The API will use n_components but under the hood we'll match the Fortran naming
    # TODO: Maybe rename n_components to num_comps in the config dataclass?
    num_comps = config.n_components
    num_models = config.n_models
    num_mix = config.n_mixtures
    # !-------------------- ALLOCATE VARIABLES ---------------------
    print("Allocating variables ...")

    # !------------------- INITIALIZE VARIABLES ----------------------
    # print *, myrank+1, ': Initializing variables ...'; call flush(6);
    # if (seg_rank == 0) then
    print("Initializing variables ...")

    assert_allclose(state.gm.sum(), 1.0)
    # load_alpha:
    state.alpha[:, :num_mix] = 1.0 / num_mix
    # load_mu:
    mu_values = torch.arange(num_mix) - (num_mix - 1) / 2
    state.mu[:, :] = mu_values[None, :]
    if not fix_init:
        mutmp = MUTMP.T
        state.mu[:, :num_mix] = state.mu[:, :num_mix] + 0.05 * (1.0 - 2.0 * mutmp)
    # load_beta:
    if fix_init:
        raise NotImplementedError()
    else:
        state.sbeta[:, :num_mix] = 1.0 + 0.1 * (0.5 - sbetatmp)
    # load_c:
    state.c.fill_(0.0)
    
    # load_A:
    for h, _ in enumerate(range(num_models), start=1):
        h_index = h - 1
        comp_slice = get_component_slice(h=h, n_components=num_comps)
        state.A[:, comp_slice] = 0.01 * (0.5 - WTMP)        
        idx = torch.arange(num_comps)
        cols = h_index * num_comps + idx
        state.A[idx, cols] = 1.0
        Anrmk = torch.linalg.norm(state.A[:, cols], dim=0)
        state.A[:, cols] /= Anrmk   
    # end load_A
    
    iterating = True if "iter" in locals() else False
    W, wc = get_unmixing_matrices(
        iterating=iterating,
        c=state.c,
        A=state.A,
        comp_slice=comp_slice,
        W=state.W,
        num_models=num_models,
    )
    assert W.dtype == torch.float64
    state.W = W.clone()
    del W # safe guard against accidental use of W instead of state.W


    # !-------------------- Determine optimal block size -------------------
    print(f"1: block size = {config.batch_size}")
    
    # !XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX main loop XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    print(f"1 : entering the main loop ...")
    with torch.no_grad():
        state, LL = optimize(
            X=X,
            sldet=sldet.item(),
            wc=wc,
            config=config,
            state=state,
        )
    # Convert Tensors to numpy arrays for output
    state_dict = state.to_numpy()
    LL = LL.numpy()
    return state_dict, LL
    


@line_profiler.profile
def optimize(
        *,
        X: DataTensor2D,
        sldet: float,
        wc: ParamsModelTensor,
        config: AmicaConfig,
        state: AmicaState,
):
    """Main optimization loop for AMICA."""
    leave = False
    iter = 1
    numrej = 0
    N1 = config.batch_size
    pdftype = config.pdftype
    do_reject = do_reject = config.do_reject
    do_newton = config.do_newton
    newt_start = config.newt_start
    assert newt_start == 50
    

    # Initialize accumulators container
    accumulators = initialize_accumulators(config)
    # We allocate these separately.
    Dsum = torch.zeros(config.n_models, dtype=torch.float64)
    Dsign = torch.zeros(config.n_models, dtype=torch.float64)
    loglik = torch.zeros((X.shape[1],), dtype=torch.float64)  # per sample log likelihood
    LL = torch.zeros(max(1, config.max_iter), dtype=torch.float64)  # Log likelihood history

    min_dll = config.tol
    numincs = 0  # number of consecutive iterations where likelihood increased by less than tol/min_dll
    lrate = config.lrate
    rholrate = config.rholrate
    c_start = time.time()
    c1 = time.time()
    while iter <= config.max_iter:
        # ============================== Subsection ====================================
        # === Update the unmixing matrices and compute the determinants ===
        # ===============================================================================
        
        # !----- get determinants
        loglik.fill_(0.0)
        metrics = IterationMetrics(
            iter=iter,
            lrate=lrate,
            rholrate=rholrate,
        )
        iter = metrics.iter


        for h, _ in enumerate(range(config.n_models), start=1):
            h_index = h - 1
            # The Fortran code computed log|det(W)| indirectly via QR factorization
            # We use slogdet on the original unmixing matrix to get sign and log|det|
            sign, logabsdet = compute_sign_log_determinant(
                unmixing_matrix=state.W[:, :, h_index],
                minlog=minlog,
            )
            Dsum[h_index] = logabsdet
            Dsign[h_index] = sign                 

        nw = config.n_components
        accumulators.reset()
        # !--------- loop over the segments ----------
        if do_reject:
            raise NotImplementedError()
        else:
            pass
        # !--------- loop over the blocks ----------
        '''
        # In Fortran, the OMP parallel region would start before the lines below.
        # !$OMP PARALLEL DEFAULT(SHARED) &
        # ...
        # !print *, myrank+1, thrdnum+1, ': Inside openmp code ... '; call flush(6)
        '''

        # -- 0. Baseline terms for per-sample model log-likelihood --
        initial = get_initial_model_log_likelihood(
                unmixing_logdet=Dsum[h_index],
                whitening_logdet=sldet,
                model_weight=state.gm[h_index],
        )
        
        #=============================== Subsection =====================================
        # === Begin chunk loop ===
        # ===============================================================================
        batch_loader = BatchLoader(X, axis=-1, batch_size=N1)
        for batch_idx, (data_batch, batch_indices) in enumerate(batch_loader):
            for h, _ in enumerate(range(config.n_models), start=1):
                comp_slice = get_component_slice(h, config.n_components)
                h_index = h - 1
                
                # ===========================================================================
                #                       Expectation Step (E-step)
                # ===========================================================================

                # 1. --- Compute source pre-activations
                # !--- get b
                b = compute_preactivations(
                    X=data_batch,
                    unmixing_matrix=state.W[:, :, h_index],
                    bias=wc[:, h_index],
                    do_reject=config.do_reject,
                )
                
                # 2. --- Source densities, and per-sample mixture log-densities (logits) ---
                y, z = _compute_source_densities(
                    pdftype=config.pdftype,
                    b=b,
                    sbeta=state.sbeta,
                    mu=state.mu,
                    alpha=state.alpha,
                    rho=state.rho,
                    comp_slice=comp_slice,
                    )
                z0 = z  # log densities (alias for clarity with Fortran code)

                # 3. --- Aggregate mixture logits into per-sample model log likelihoods
                modloglik = torch.full(
                    size=(data_batch.shape[1], config.n_models),
                    fill_value=initial,
                    dtype=config.dtype,
                    )
                compute_model_loglikelihood_per_sample(
                    log_densities=z0,
                    out_modloglik=modloglik[:, h_index],
                )
                
                # 4. -- Responsibilities within each component ---
                # !--- get normalized z
                z = compute_mixture_responsibilities(log_densities=z0, inplace=True)
                z0 = None  # guard against use of stale name
            # end do (h)

            # 5. --- Across-model Responsibilities and Total Log-Likelihood ---
            loglik[batch_indices] = compute_total_loglikelihood_per_sample(
                modloglik=modloglik,
                out_loglik=loglik[batch_indices]
            )

            if do_reject:
                raise NotImplementedError()
            else:    
                # 6. --- Responsibilities for each model ---
                v = compute_model_responsibilities(
                    modloglik=modloglik,
                    )
            # ================================ M-STEP ===================================
            # === Maximization-step: Parameter accumulators ===
            # - Update parameters based on current responsibilities
            # - Update unmixing matrices with gradient ascent and optional Newton-Raphson
            # ===========================================================================
            
            # !--- get g, u, ufp
            for h, _ in enumerate(range(config.n_models), start=1):
                comp_slice = get_component_slice(h, config.n_components)
                h_index = h - 1
                #--------------------------FORTRAN CODE-------------------------
                # vsum = sum( v(bstrt:bstp,h) )
                # dgm_numer_tmp(h) = dgm_numer_tmp(h) + vsum 
                #---------------------------------------------------------------
                v_h = v[:, h_index] #  select responsibilities for this model
                vsum = v_h.sum()

                u = compute_weighted_responsibilities(
                    mixture_responsibilities=z,
                    model_responsibilities=v_h,
                    single_model=(config.n_models == 1),
                )
                usum = u.sum(dim=0)  # shape: (nw, num_mix)

                fp = compute_source_scores(
                    pdftype=pdftype,
                    y=y,
                    rho=state.rho,
                    comp_slice=comp_slice,
                )
                # --- Vectorized calculation of ufp and g update ---

                ufp, g = accumulate_scores(
                    scores=fp,
                    responsibilities=u,
                    scale_params=state.sbeta,
                    comp_slice=comp_slice,
                )

                # --- Stochastic Gradient Descent accumulators ---
                # gm (model weights)
                accumulators.dgm_numer[h_index] += vsum
                # c (bias)  
                accumulate_c_stats(
                    X=data_batch,
                    model_responsibilities=v_h,
                    vsum=vsum,
                    out_numer=accumulators.dc_numer[:, h_index],
                    out_denom=accumulators.dc_denom[:, h_index],
                )
                # Alpha (mixture weights)
                accumulate_alpha_stats(
                    usum=usum,
                    vsum=vsum,
                    out_numer=accumulators.dalpha_numer[comp_slice, :],
                    out_denom=accumulators.dalpha_denom[comp_slice, :],
                )
                # Mu (location)
                accumulate_mu_stats(
                    ufp=ufp,
                    y=y,
                    sbeta=state.sbeta[comp_slice, :],
                    rho=state.rho[comp_slice, :],
                    out_numer=accumulators.dmu_numer[comp_slice, :],
                    out_denom=accumulators.dmu_denom[comp_slice, :],
                )
                # Beta (scale/precision)
                accumulate_beta_stats(
                    usum=usum,
                    rho=state.rho[comp_slice, :],
                    ufp=ufp,
                    y=y,
                    out_numer=accumulators.dbeta_numer[comp_slice, :],
                    out_denom=accumulators.dbeta_denom[comp_slice, :],
                )
                # Rho (shape parameter of generalized Gaussian)
                accumulate_rho_stats(
                    y=y,
                    rho=state.rho[comp_slice, :],
                    u=u,
                     usum=usum,
                    epsdble=epsdble,
                    out_numer=accumulators.drho_numer[comp_slice, :],
                    out_denom=accumulators.drho_denom[comp_slice, :],
                )
                # --- Newton-Raphson accumulators ---
                if do_newton and iter >= newt_start:
                    if iter == 50 and batch_indices.start == 0:
                        assert torch.all(accumulators.newton.dkappa_numer == 0.0)
                        assert torch.all(accumulators.newton.dkappa_denom == 0.0)
                    # NOTE: Fortran computes dsigma_* for in all iters, but thats unnecessary
                    # Sigma^2 accumulators (noise variance)
                    accumulate_sigma2_stats(
                        model_responsibilities=v_h,
                        source_estimates=b,
                        vsum=vsum,
                        out_numer=accumulators.newton.dsigma2_numer[:, h_index],
                        out_denom=accumulators.newton.dsigma2_denom[:, h_index],
                    )
                    # Kappa accumulators (curvature terms for A)
                    accumulate_kappa_stats(
                        ufp=ufp,
                        fp=fp,
                        sbeta=state.sbeta[comp_slice, :],
                        usum=usum,
                        out_numer=accumulators.newton.dkappa_numer[:, :, h_index],
                        out_denom=accumulators.newton.dkappa_denom[:, :, h_index],
                    )
                    # Lambda accumulators (nonlinearity shape parameter)
                    accumulate_lambda_stats(
                        fp=fp,
                        y=y,
                        u=u,
                        usum=usum,
                        out_numer=accumulators.newton.dlambda_numer[:, :, h_index],
                        out_denom=accumulators.newton.dlambda_denom[:, :, h_index],
                    )
                    # (dbar)Alpha accumulators
                    accumulators.newton.dbaralpha_numer[:, :, h_index] += usum
                    accumulators.newton.dbaralpha_denom[:,:, h_index] += vsum
                # end if (do_newton and iter >= newt_start)
                elif not do_newton and iter >= newt_start:
                    raise NotImplementedError()

                # if (print_debug .and. (blk == 1) .and. (thrdnum == 0)) then
                # if update_A:
                #--------------------------FORTRAN CODE-------------------------
                # call DSCAL(nw*nw,dble(0.0),Wtmp2(:,:,thrdnum+1),1)   
                # call DGEMM('T','N',nw,nw,tblksize,dble(1.0),g(bstrt:bstp,:),tblksize,b(bstrt:bstp,:,h),tblksize, &
                #            dble(1.0),Wtmp2(:,:,thrdnum+1),nw)
                # call DAXPY(nw*nw,dble(1.0),Wtmp2(:,:,thrdnum+1),1,dWtmp(:,:,h),1)
                #---------------------------------------------------------------
                accumulators.dA[:, :, h - 1] += torch.matmul(g.T, b)
            # end do (h)
        # end do (blk)'

        # In Fortran, the OMP parallel region is closed here
        # !$OMP END PARALLEL
        
        likelihood, ndtmpsum, no_newt = accum_updates_and_likelihood(
            config=config,
            accumulators=accumulators,
            state=state,
            total_LL=loglik.sum(),
            iter=iter
        )
        metrics.loglik = likelihood
        metrics.ndtmpsum = ndtmpsum
        metrics.no_newt = no_newt
        # return accumulators, metrics

        # ==============================================================================
        ndtmpsum = metrics.ndtmpsum
        LL[iter - 1] = metrics.loglik
        # init
        startover = False
        numdecs = 0
             
        # !----- display log likelihood of data
        # if (seg_rank == 0) then
        c2 = time.time()
        t0 = c2 - c1
        #  if (mod(iter,outstep) == 0) then

        if (iter % outstep) == 0:
            print(
                f"Iteration {iter}, lrate = {lrate:.3f}, LL = {LL[iter - 1]:.3f}, "
                f"nd = {ndtmpsum:.3f}, D = {Dsum.max():.3f} {Dsum.min():.3f} "
                f"took {t0:.2f} seconds"
                )
            c1 = time.time()

        # !----- check whether likelihood is increasing
        # if (seg_rank == 0) then
        # ! if we get a NaN early, try to reinitialize and startover a few times 
        if (iter <= restartiter and torch.isnan(LL[iter - 1])):
            if numrestarts > maxrestarts:
                leave = True
                raise RuntimeError()
            else:
                raise NotImplementedError()
        # end if
        if iter == 2:
            assert not torch.isnan(LL[iter - 1])
            assert not (LL[iter - 1] < LL[iter - 2])
        if iter > 1:
            if torch.isnan(LL[iter - 1]) and iter > restartiter:
                leave = True
                raise RuntimeError(f"Got NaN! Exiting")
            if (LL[iter - 1] < LL[iter - 2]):
                assert 1 == 0
                print("Likelihood decreasing!")
                if (lrate < minlrate) or (ndtmpsum <= min_nd):
                    leave = True
                    print("minimum change threshold met, exiting loop")
                else:
                    lrate *= lratefact
                    rholrate *= rholratefact
                    numdecs += 1
                    if numdecs >= maxdecs:
                        raise NotImplementedError()
                        lrate0 *= lrate0 * lratefact
                        if iter == 2:
                            assert 1 == 0
                        if iter > config.newt_start:
                            raise NotImplementedError()
                            rholrate0 *= rholratefact
                        if config.do_newton and iter > config.newt_start:
                            print("Reducing maximum Newton lrate")
                            newtrate *= lratefact
                            assert 1 == 0 # stop to check that value
                        numdecs = 0
                    # end if (numdecs >= maxdecs)
                # end if (lrate vs minlrate)
            # end if LL
            if use_min_dll:
                if (LL[iter - 1] - LL[iter - 2]) < min_dll:
                    numincs += 1
                    if numincs > maxincs:
                        leave = True
                        print(
                            f"Exiting because likelihood increasing by less than {min_dll} "
                            f"for more than {maxincs} iterations ..."
                            )
                else:
                    numincs = 0
                if iter == 2:
                    assert numincs == 0
            else:
                raise NotImplementedError() # pragma no cover
            if use_grad_norm:
                if ndtmpsum < min_nd:
                    leave = True
                    print(
                        f"Exiting because norm of weight gradient less than {min_nd:.6f} ... "
                    )
                    assert 1 == 0
                if iter == 2:
                    assert leave is False
            else:
                raise NotImplementedError() # pragma no cover
        # end if (iter > 1)
        if config.do_newton and (iter == config.newt_start):
            print("Starting Newton ... setting numdecs to 0")

        # call MPI_BCAST(leave,1,MPI_LOGICAL,0,seg_comm,ierr)
        # call MPI_BCAST(startover,1,MPI_LOGICAL,0,seg_comm,ierr)

        if leave:
            return state, LL
        if startover:
            raise NotImplementedError()
        else:
            # !----- do accumulators: gm, alpha, mu, sbeta, rho, W
            # the updated lrate & rholrate for the next iteration
            lrate, rholrate, state, wc = update_params(
                config=config,
                state=state,
                accumulators=accumulators,
                metrics=metrics,
                wc=wc,
            )
            # !----- reject data
            if (
                config.do_reject
                and (maxrej > 0)
                and (
                    iter == rejstart
                    or (max(1, iter-rejstart) % rejint == 0 and numrej < maxrej)
                )
            ):
                raise NotImplementedError()            
            iter += 1
        # end if/else
    # end while
    c_end = time.time()
    print(f"Finished in {c_end - c_start:.2f} seconds")
    return state, LL



def get_unmixing_matrices(
        *,
        iterating,
        c,
        A,
        comp_slice: slice,
        W,
        num_models,
        ):
    """Get unmixing matrices for AMICA."""

    wc = torch.zeros_like(c)
    for h, _ in enumerate(range(num_models), start=1):

        #--------------------------FORTRAN CODE-------------------------
        # call DCOPY(nw*nw,A(:,comp_list(:,h)),1,W(:,:,h),1)
        #---------------------------------------------------------------
        # TODO: assign on the fly without passing in W
        W[:, :, h - 1] = A[:, comp_slice].clone()


        #--------------------------FORTRAN CODE-------------------------
        # call DGETRF(nw,nw,W(:,:,h),nw,ipivnw,info)
        # call DGETRI(nw,W(:,:,h),nw,ipivnw,work,lwork,info)
        #---------------------------------------------------------------
        try:
            W[:, :, h - 1] = torch.linalg.inv(W[:, :, h - 1])
        except RuntimeError as e:
            # This issue would originate with matrix A
            # we should review the code and provide a more user friendly error message
            # if A is singular. e.g. the "weights matrix is singular or something"
            print(f"Matrix W[:,:,{h-1}] is singular!")
            raise e

        #--------------------------FORTRAN CODE-------------------------
        # call DGEMV('N',nw,nw,dble(1.0),W(:,:,h),nw,c(:,h),1,dble(0.0),wc(:,h),1)
        #---------------------------------------------------------------
        wc[:, h - 1] = W[:, :, h - 1] @ c[:, h - 1]

    return W, wc

def get_seg_list(raw):
    """This is a temporary function that somehwat mirrors the Fortran get_seg_list"""
    blocks_in_sample = raw.n_times  # field_dim
    num_samples = 1  # num_files
    all_blks = blocks_in_sample * num_samples
    # We'll stop here for now. and port more of the Fortran function as we need it.
    return blocks_in_sample, num_samples, all_blks


def get_accumulators_and_likelihood(
    X,
    *,
    config,
    state,
    accumulators,
    metrics,
    work,
    sldet,
    Dsum,
    wc,
):
    """Get accumulators and likelihood for AMICA.
    
    Purpose:
        - E-step: compute per-model/per-component log-likelihoods and responsibilities.
        - M-step: accumulate sufficient statistics (update numerators/denominators)
        for parameters like `A`, `mu`, `sbeta`, and `rho`.
    Notes
    - This function mirrors the original Fortran implementation. Fortran reference
        comment blocks are kept verbatim alongside the equivalent Python.
    """
    pass


def compute_model_e_step(
        *,
        X,
        config,
        alpha,
        sbeta,
        gm,
        mu,
        rho,
        W,
        buffers,
        model_index,
        comp_slice,
        Dsum,
        sldet,
        wc,
):
    """Per-Model expectation (E) step.
    
    For a single model index, compute the posterior responsiblities
    (z), latent-variable stats (y), per-sample model evidence (Ptmp),
    source estimates (b).

    Parameters
    ----------
    X : np.ndarray
        The input data array of shape (n_samples, n_features). This array is not
        modified.
    config : AmicaConfig
        Configuration object containing model parameters. The following attributes are used:
        - n_components: Number of components (sources).
        - n_mixtures: Number of mixtures.
        - pdftype: Probability density function type.
        - do_reject: Boolean indicating whether to perform rejection.
    buffers : AmicaWorkspace
        Workspace buffers for intermediate computations. The following buffers are modified in-place:
        - b: Per-sample, per-component source estimates, of shape (n_samples, n_features, n_components).
        - y: Scaled sources of shape (n_samples, n_features, n_mixtures, n_components).
        - z: Per-mixture log-densities, of shape (n_samples, n_features, n_mixtures, n_components).
        - Ptmp: Buffer for log-likelihood computations of shape (n_samples, n_components).
    alpha : np.ndarray
        Mixture weights of shape (n_mixtures, n_features). This array is not modified.
    sbeta : np.ndarray
        Scale parameters of shape (n_mixtures, n_features). This array is not modified.
    gm : np.ndarray
        Mixing proportions of shape (n_features,). This array is not modified.
    mu : np.ndarray
        Location parameters of shape (n_mixtures, n_features). This array is not modified.
    rho : np.ndarray
        Shape parameters of shape (n_mixtures, n_features). This array is not modified.
    W : np.ndarray
        Unmixing matrix of shape (n_features, n_features, n_components). 
    model_index : int
        Index of the current model being processed. This is used to index arrays
        with a dimension for models.
    comp_slice : slice
        Slice object containing component indices for the current model.
    Dsum : np.ndarray
        Precomputed sum of squared data projections for each component values for
        the current model of shape (n_components,). This array is not modified.
    sldet : float
        Precomputed Log-determinant of the unmixing matrix (W) for the current model. Not modified.
    wc : np.ndarray
        Precomputed weight correction factors for the current model
        shape (n_features,). This array is not modified.
    
    Returns
    -------
    This function modifies the `b`, `y`, `z`, and `Ptmp` buffers in-place,
    and returns the modified arrays.
    """
    pass


def compute_sign_log_determinant(
        *,
        unmixing_matrix: WeightsArray,
        minlog: float = -1500,
        mode: Literal["strict", "fallback"] = "strict", 
) -> tuple[Literal[-1, 1], float]:
    """Compute the sign and log-determinant of the unmixing matrix for a single model.
    
    Parameters
    ----------
    unmixing_matrix: array, shape (n_components, n_features)
        The unmixing matrix W for a single model h (i.e. a 2D slice of state.W).
    minlog: float
        Minimum log value: log absolute determinant to return if the computed
        log-determinant is zero. Default is -1500, but currently if the computed
        log-determinant is zero, an error is raised instead.
    mode: str
        Mode for handling cases where the computed log-determinant is zero.
        default is "strict", Options are:
        - "strict": Raise a ValueError if the log-determinant is zero.
        - "fallback": Issue a warning and then set the log-determinant to minlog
            (default: -1500) and set sign to -1.

    Returns
    -------
    sign: {-1, 1}
        The sign of the determinant (+1 or -1). In "fallback" mode, sign is set to −1 if
        the determinant is zero, to maintain the invariant that sign ∈ {−1, 1} (never 0).
    logabsdet: float
        The (natural) log-determinant of the unmixing matrix. In "fallback" mode, this is
        set to minlog (default: -1500).
    """
    #--------------------------------FORTRAN CODE------------------------------
    #    call DCOPY(nw*nw,W(:,:,h),1,Wtmp,1)
    # ....
    #    call DGEQRF(nw,nw,Wtmp,nw,wr,work,lwork,info)
    # ...
    # Dtemp(h) = dble(0.0)
    # ...
    #   Dtemp(h) = Dtemp(h) + log(abs(Wtmp(i,i)))
    # ------------------------------------------------------------------------
    # Alias for clarity with Fortran code
    W = unmixing_matrix
    sign, logabsdet = np.linalg.slogdet(W)
    # TODO: slogdet requires a square unmixing matrix. Confirm that AMICA gaurantees this
    if logabsdet == -np.inf or sign == 0:  # Model fit has collapsed.
        msg = (
                "Unmixing matrix (W) is singular (determinant = 0)\n\n"
                "Details:\n"
                f"- shape of W: {W.shape}\n"
                f"- sign={sign}, log|det|={logabsdet}\n\n"
                "Things to try:\n"
                "- Check that your input data is rank-sufficient\n"
                "- Reduce the number of components\n"
        )
        if mode == "strict":
            # By default Let's raise an error until we can test this case properly
            raise ValueError(msg)
        else:
            print(msg)
            print(f"Setting log-determinant to {minlog} and sign to -1")
            # fallback values (numerical hack to let training continue)
            logabsdet = minlog
            sign = -1  # matches dsign = 1 if det > 0 else -1 in Fortran
    return sign, logabsdet


def get_initial_model_log_likelihood(
        *,
        unmixing_logdet: float,
        whitening_logdet: float,
        model_weight: float,
) -> float:
    """
    Initialize the per-sample model log-likelihood with baseline terms.
    
    Parameters
    ----------
    unmixing_logdet : float
        The log-determinant of the unmixing matrix (W) for this model.
    whitening_logdet : float
        The log-determinant of the sphering/whitening transform (S),
        computed from the input data's whitening/sphering matrix. e.g. It's computed
        as -0.5 ∑ log(λ_i) where λ_i are covariance eigenvalues.
    model_weight : float
        The mixture proportion (prior probability) for this model.

    Returns
    -------
    initial_modloglik : float
        A scalar baseline log-likelihood value. This should be broadcast across all
        samples of the model log-likelihood array at the call site.
    
    Notes
    -----
    - The Jacobian from x → u is |det(W S)|, so log|det(W S)| = log|det(W)| +
    log|det(S)| = Dsum[h] + sldet.
    - S is positive-definite with full-rank whitening, so sldet has no sign
    issue
    """
    whitening_logdet = torch.as_tensor(whitening_logdet, dtype=torch.float64)
    unmixing_logdet = torch.as_tensor(unmixing_logdet, dtype=torch.float64)
    if model_weight <= 0:
        raise ValueError(f"model_weight must be > 0, got {model_weight}")  # pragma no cover noqa: E501
    #--------------------------FORTRAN CODE-------------------------
    # Ptmp(bstrt:bstp,h) = Dsum(h) + log(gm(h)) + sldet
    #---------------------------------------------------------------
    initial_modloglik = unmixing_logdet + torch.log(model_weight) + whitening_logdet
    return initial_modloglik


def compute_preactivations(
        *,
        X: DataArray2D,  # (n_features, n_samples)
        unmixing_matrix: WeightsArray,  # (n_components, n_features)
        bias: ComponentsVector,  # (n_components,)
        do_reject=False, 
        out_activations: Optional[SourceArray2D] = None,  # (n_samples, n_components)
) -> SourceArray2D:
    """Compute source pre-activations b[t, :] = X_t^T @ W^T - wc for model h.
    
    Parameters
    ----------
    X : array, shape (n_features, batch_size)
        Data matrix. Can be the entire input data or a chunk. Not modified.
    unmixing_matrix : array, shape (n_components, n_features)
        Unmixing matrix weights (W) for a single model h, that maps data to sources.
    bias : array, shape (n_components,)
        Weight correction (wc) vector for model h e.g. wc[:, h_index]
    do_reject : bool
        Whether to perform rejection. Currently not implemented.
    
    Returns
    -------
    b : array, shape (T, N)
        Pre-activations matrix of shape (n_samples, n_features)
    """
    # if update_c and update_A:
    #--------------------------FORTRAN CODE-------------------------
    # call DSCAL(nw*tblksize,dble(0.0),b(bstrt:bstp,:,h),1)
    #---------------------------------------------------------------
    # Alias terms for clarity with Fortran code
    dataseg = X
    W = unmixing_matrix
    wc = bias
    assert wc.ndim == 1, f"wc must be 1D, got {wc.ndim}D"
    assert W.ndim == 2, f"W must be 2D, got {W.ndim}D"
    assert dataseg.ndim == 2, f"dataseg must be 2D, got {dataseg.ndim}D"

    if do_reject:
        #--------------------------FORTRAN CODE-------------------------
        # call DGEMM('T','T',tblksize,nw,nw,dble(1.0),dataseg(seg)%data(:,dataseg(seg)%goodinds(xstrt:xstp)),nx, &
        #       W(:,:,h),nw,dble(1.0),b(bstrt:bstp,:,h),tblksize)
        #---------------------------------------------------------------
        raise NotImplementedError("do_reject (rejecting bad data) is not implemented.")
    else:
        # Multiply the transpose of the data w/ the transpose of the unmixing matrix
        #--------------------------FORTRAN CODE-------------------------
        # call DGEMM('T','T',tblksize,nw,nw,dble(1.0),dataseg(seg)%data(:,xstrt:xstp),nx,W(:,:,h),nw,dble(1.0), &
        #    b(bstrt:bstp,:,h),tblksize)
        #---------------------------------------------------------------
        
        # Matrix multiplication to get pre-activations
        # This is equivalent to (f=features, t=samples, c=components):
        # Same as np.einsum("ft,cf->tc", dataseg, W)
        b = torch.matmul(dataseg.T, W.T)
    # end else
    # Subtract the weight correction factor
    b -= wc
    return b


def _compute_source_densities(
        *,
        pdftype: int,
        b: SourceArray2D,
        sbeta: ParamsArray,
        mu: ParamsArray,
        alpha: ParamsArray,
        rho: ParamsArray,
        comp_slice: slice, # TODO: pass in the pre-indexed arrays instead
        ) -> Tuple[SourceArray3D, SourceArray3D]:
    """Calculate scaled sources (y) and per-mixture log-densities (logits).

     Compute logits = log alpha + log p(y) for each component and mixture. Default to
     generalized Gaussian, then overwrite Laplacian/Gaussian positions using masks.
     This way is faster since vast majority of values are generalized Gaussian.
    
    Parameters
    ----------
    pdftype : int
        Probability density function type. Currently, only 0 (Gaussian) is supported.
    b : np.ndarray
        Per-sample, per-component source estimates, of shape (n_samples, n_components).
        Not modified.
    sbeta : np.ndarray
        Scale parameters. Shape (n_components, n_mixtures).
        Not modified.
    mu : np.ndarray
        Location parameters. Shape (n_components, n_mixtures). Not modified.
    alpha : np.ndarray
        Mixture weights. Shape (n_components, n_mixtures). Not modified.
    rho : np.ndarray
        Shape parameters. Shape (n_components, n_mixtures). Not modified.
    comp_slice : slice
        slice containing component indices for the current model.
    out_sources : np.ndarray
        Buffer to write scaled sources into. Shape (n_samples, n_components, n_mixtures).
        This array is modified in-place. This is the `y` array in the Fortran code.
    out_logits : np.ndarray
        Buffer to write per-mixture log-densities into. Shape (n_samples, n_components, n_mixtures).
        This array is modified in-place. This is the `z0` array in the Fortran code.

    Returns
    -------
    scaled_sources : np.ndarray
        The modified y array containing scaled sources (n_samples, n_components, n_mixtures).
    log_densities : np.ndarray
        unnormalized log-probabilies (i.e. posteriors) per mixtures. this is the
        out_z0 array modified in-place (n_samples, n_components, n_mixtures).
    
    Notes
    -----
    logits/log_densities == z0 in Fortran code.

    For each sample t, component i, mixture j, evaluate the log of the source
    distribution density: How likely each individual source sample is under each
    mixture component.
    """
    N1 = b.shape[0]
    nw = comp_slice.stop - comp_slice.start
    num_mix = alpha.shape[1]
    
    # Shape assertions for new dimension standard
    # These parameters are the full arrays, not indexed yet
    assert alpha.shape[0] >= nw, f"alpha.shape[0]={alpha.shape[0]} must be >= nw={nw}"
    assert alpha.shape[1] == num_mix, f"alpha.shape[1]={alpha.shape[1]} != num_mix={num_mix}"
    assert sbeta.shape == alpha.shape, f"sbeta shape {sbeta.shape} != alpha shape {alpha.shape}"
    assert mu.shape == alpha.shape, f"mu shape {mu.shape} != alpha shape {alpha.shape}"
    assert rho.shape == alpha.shape, f"rho shape {rho.shape} != alpha shape {alpha.shape}"
    assert b.shape == (N1, nw), f"b shape {b.shape} != (N1={N1}, nw={nw})"
    
    # We have 3 possible log-probability functions
    def generalized_gaussian_logprob(sources, log_alpha, log_sbeta, rho):
        """log p(y) = log(alpha) + log(sbeta) - |y|^rho - log( Gamma(1+1/rho) ) + log(2)"""
        # log(|y|)
        out_logits = torch.abs(sources)
        torch.log(out_logits, out=out_logits)
        # |y|^rho
        torch.exp(rho * out_logits, out=out_logits)
        # log(alpha) + log(sbeta) - |y|^rho
        torch.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        # gammaln(1 + 1/rho)
        gamma_log = torch.special.gammaln(1.0 + 1.0 / rho)
        # penalty term: log(alpha) + log(sbeta) - |y|^rho - gammaln(1 + 1/rho) + log(2)
        torch.subtract(out_logits, gamma_log + LOG_2, out=out_logits)
        return out_logits

    def laplacian_logprob(sources, log_alpha, log_sbeta, out_logits):
        """log p(y) = log(alpha) + log(sbeta) - |y| - log(2)"""
        torch.abs(sources, out=out_logits)
        torch.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        torch.subtract(out_logits, LOG_2, out=out_logits)
        return out_logits

    def gaussian_logprob(sources, log_alpha, log_sbeta, out_logits):
        """log p(y) = log(alpha) + log(sbeta) - y^2 - log(sqrt(pi))"""
        torch.square(sources, out=out_logits)
        torch.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        torch.subtract(out_logits, LOG_SQRT_PI, out=out_logits)
        return out_logits
  
    # ---------------------------FORTRAN CODE-------------------------
    # !--- get y z
    # do i = 1,nw
    # !--- get probability
    # select case (pdtype(i,h))
    #-----------------------------------------------------------------
    if pdftype == 0: # Gaussian
        #--------------------------FORTRAN CODE-------------------------
        # y(bstrt:bstp,i,j,h) = sbeta(j,comp_list(i,h)) * ( b(bstrt:bstp,i,h) - mu(j,comp_list(i,h)) )
        #---------------------------------------------------------------
        # 1. Select the components for this model.
        alpha_h = alpha[comp_slice, :]
        rho_h = rho[comp_slice, :] # All mixtures, components for this model
        sbeta_h = sbeta[comp_slice, :]      # Shape: (nw, num_mix)
        mu_h = mu[comp_slice, :]            # Shape: (nw, num_mix)
        
        # 1. Center and scale the source estimates (In-place)
        out_sources = torch.subtract(b[:, :, None], mu_h[None, :, :])
        torch.multiply(sbeta_h, out_sources, out=out_sources)
        
        #------------------Mixture Log-Likelihood for each component----------------

        #--------------------------FORTRAN CODE-------------------------
        # if (rho(j,comp_list(i,h)) == dble(1.0)) then
        # else if (rho(j,comp_list(i,h)) == dble(2.0)) then
        # z0(bstrt:bstp,j) = log(alpha(j,comp_list(i,h))) + ...
        #---------------------------------------------------------------
        # Precompute logs (reused in all 3 logprob functions)
        log_mixture_weights = torch.log(alpha_h)  # shape: (nw, num_mix)
        log_scales = torch.log(sbeta_h)           # shape: (nw, num_mix)

        # Masks: Laplacian (rho==1), Gaussian (rho==2); generalized Gaussian otherwise
        lap_mask = (torch.isclose(rho_h, torch.tensor(1.0, dtype=torch.float64), atol=1e-12))
        gau_mask = (torch.isclose(rho_h, torch.tensor(2.0, dtype=torch.float64), atol=1e-12))

        # Default: generalized Gaussian log-prob + log mixture weight 
        # This is all or the vast majority of values, so just compute gen gau over all
        # and then overwrite lap/gau where needed using a small loop
        out_logits = generalized_gaussian_logprob(
            sources=out_sources,
            log_alpha=log_mixture_weights,
            log_sbeta=log_scales,
            rho=rho_h,
        )
        assert out_logits.shape == (N1, nw, num_mix)
        # Overwrite with Laplacian/Gaussian log-prob + log mixture weight where needed
        # This is usually a small loop, and ensures we get a view of the arrays
        if lap_mask.any():
            for i, j in zip(*lap_mask.nonzero(as_tuple=True)):
                out_logits[:, i, j] = laplacian_logprob(
                    sources=out_sources[:, i, j],
                    log_alpha=log_mixture_weights[i, j],
                    log_sbeta=log_scales[i, j],
                    out_logits=out_logits[:, i, j]
                )
        if gau_mask.any():
            for i, j in zip(*gau_mask.nonzero(as_tuple=True)):
                out_logits[:, i, j] = gaussian_logprob(
                    sources=out_sources[:, i, j],
                    log_alpha=log_mixture_weights[i, j],
                    log_sbeta=log_scales[i, j],
                    out_logits=out_logits[:, i, j]
                )
        # end if lap_mask/gau_mask.any()
    elif pdftype == 1:
        raise NotImplementedError()
    elif pdftype == 2:
        raise NotImplementedError()
    elif pdftype == 3:
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid pdftype {pdftype}. Only pdftype=0 (Gaussian) is supported.")
    # end select
    # !--- end for j
    return out_sources, out_logits


def compute_model_loglikelihood_per_sample(
        *,
        log_densities: SourceArray3D,
        out_modloglik: Optional[SamplesVector] = None,
):
    """Compute the per-sample log-likelihood for a single model h.

    Computes Logsumexp across mixtures for each component, then sums across components
    to get the per-sample log-likelihood for this model.
    
    Parameters
    ----------
    log_densities : array, shape (N1, nw, num_mix)
        Per-sample, per-component, per-mixture log-densities.
    out_modloglik : array, shape (N1,)
        Output array for per-sample log-likelihood for this model. If None,
        a new array is allocated. This array is mutated in-place.
    """
    assert log_densities.ndim == 3, f"log_densities must be 3D, got {log_densities.ndim}D"
    N1 = log_densities.shape[0]
    nw = log_densities.shape[1]
    if out_modloglik is None:
        out_modloglik = torch.zeros((N1,), dtype=torch.float64)
    assert out_modloglik.shape == (N1,)
    # Alias for clarity with Fortran code
    z0 = log_densities

    #--------------------------FORTRAN CODE-------------------------
    # Pmax(bstrt:bstp) = maxval(z0(bstrt:bstp,:),2)
    # z(bstrt:bstp,i,j,h) = dble(1.0) / exp(tmpvec(bstrt:bstp) - z0(bstrt:bstp,j))
    #---------------------------------------------------------------
    # NOTE that the scratch array that was pass in will also be used for the g update.
    # TODO: consider keeping g and z separate once chunking is implemented
    component_loglik = torch.logsumexp(z0, dim=-1) # across mixtures
    out_modloglik += component_loglik.sum(dim=-1) # across components
    return out_modloglik



def compute_mixture_responsibilities(
        *,
        log_densities: SourceArray3D,
        inplace: Literal[True, False] = True
        ) -> SourceArray3D:
    """
    Convert per-mixture log-densities to responsibilities via softmax.
    
    Mutates log_densities in-place and returns it if inplace=True (default).

    Parameters
    ----------
    log_densities : array, shape (N1, nw, num_mix)
        Per-sample, per-component, per-mixture log-densities.
    inplace : bool, default True
        If True, mutates log_densities in-place and returns it.
        If False, returns a new array and leaves log_densities unchanged.

    Returns
    -------
    responsibilities : array, shape (N1, nw, num_mix)
        Per-sample, per-component, per-mixture responsibilities.
        Each slice [:, i, :] sums to 1 over dim=-1 (mixtures).
    """
    assert log_densities.ndim == 3, f"log_densities must be (N1, nw, num_mix), got {log_densities.shape}"
    num_mix = log_densities.shape[-1]
    if inplace:
        # Use z as workspace: log-densities mutates into responsibilities
        z = log_densities
    else:
        z = log_densities.copy()
    # fast-path: if only 1 mixture, skip softmax and set responsibilities to 1
    if num_mix == 1:
        z.fill_(1.0)
    else:
        z = torch.softmax(z, dim=-1) # across mixtures
    return z


def compute_total_loglikelihood_per_sample(
        *,
        modloglik: LikelihoodArray,
        out_loglik: Optional[SamplesVector] = None,
) -> SamplesVector:
    """
    Compute the total log-likelihood for each sample marginalized across models.
    
    Implements a stable log-sum-exp across the model dim for each sample.

    Parameters
    ----------
    modloglik : array, shape (n_samples, n_models)
        Per-sample, per-model log-likelihoods.
    out_loglik : array, shape (n_samples,)
        Output buffer. If provided, results are written in-place and returned. Must be
        float64 and length n_samples. If None, a new array is allocated.
    
    Returns
    -------
    loglik : array, shape (n_samples,)
        Total log-likelihood across models per sample.
    
    Notes
    -----
    Fortran reference:
        Pmax = maxval(Ptmp, 2)
        vtmp = sum(exp(Ptmp - Pmax), axis=1)
        P = Pmax + log(vtmp)
        dataseg(seg)%loglik(xstrt:xstp) = P(bstrt:bstp)
    """
    assert modloglik.ndim == 2, (
        f"modloglik must be 2D (n_samples, n_models), got {modloglik.ndim}D"
    )
    assert modloglik.dtype == torch.float64, (
        f"modloglik must be torch.float64, got {modloglik.dtype}"
    )
    if out_loglik is not None:
        assert out_loglik.ndim == 1, (
            f"out_loglik must be 1D, got {out_loglik.shape.ndim}D"
        )
        assert out_loglik.dtype == torch.float64, (
            f"out_loglik must be torch.float64, got {out_loglik.dtype}"
        )
        assert out_loglik.shape[0] == modloglik.shape[0], (
            f"out_loglik length {out_loglik.shape[0]} != modloglik n_samples "
            f"{modloglik.shape[0]}"
        )
    #--------------------------FORTRAN CODE-----------------------------------------
    # !print *, myrank+1,':', thrdnum+1,': getting Pmax and v ...'; call flush(6)
    # !--- get LL, v
    # Pmax(bstrt:bstp) = maxval(Ptmp(bstrt:bstp,:),2)
    # vtmp(bstrt:bstp) = dble(0.0)
    # vtmp(bstrt:bstp) = vtmp(bstrt:bstp) + exp(Ptmp(bstrt:bstp,h) - Pmax(bstrt:bstp
    # P(bstrt:bstp) = Pmax(bstrt:bstp) + log(vtmp(bstrt:bstp))
    # LLinc = sum( P(bstrt:bstp) )
    # LLtmp = LLtmp + LLinc
    #-------------------------------------------------------------------------------
    loglik = torch.logsumexp(modloglik, dim=1, out=out_loglik) # across models
    return loglik


def compute_model_responsibilities(
        *, modloglik: LikelihoodArray,
        ) -> LikelihoodArray:
    """
    Compute model responsibilities via softmax over models.

    Parameters
    ----------
    modloglik : array, shape (n_samples, n_models)
        Per-sample, per-model log-likelihoods.
    
    Returns
    -------
    responsibilities : array, shape (n_samples, n_models)
        Per-sample, per-model responsibilities (posterior probabilities).
    
    Notes
    -----
    Fortran reference:
        v(bstrt:bstp,h) = dble(1.0) / exp(P(bstrt:bstp) - Ptmp(bstrt:bstp,h))
    """
    assert modloglik.ndim == 2, (
        f"Expected 2D array (n_samples, n_models) for modloglik, got {modloglik.shape}"
    )
    num_models = modloglik.shape[1]
    assert num_models >= 1, f"modloglik must have at least one model. Got {num_models}"
    v = torch.empty_like(modloglik)
    #--------------------------FORTRAN CODE-------------------------
    # v(bstrt:bstp,h) = dble(1.0) / exp(P(bstrt:bstp) - Ptmp(bstrt:bstp,h))
    #---------------------------------------------------------------

    # fast-path: if only one model, skip softmax and set responsibilities to 1
    if num_models == 1:
        v.fill_(1.0)
    else:
        v = torch.softmax(modloglik, dim=-1) # across models
    return v


def compute_weighted_responsibilities(
        *,
        mixture_responsibilities: SourceArray3D,
        model_responsibilities: SamplesVector,
        single_model: bool = True,
) -> SourceArray3D:
    """
    Compute per-sample, per-component, per-mixture responsibilities...

    Weighted by model responsibility.

    Parameters
    ----------
    mixture_responsibilities : array, shape (n_samples, n_components, n_mixtures)
        Per-sample, per-component, per-mixture responsibilities, such that
        for each sample and component, the responsibilities sum to 1 across
        mixtures. This is `z` in the Fortran code.
    model_responsibilities : array, shape (n_samples,)
        Per-sample responsibilities for the current model. If single_model is True,
        this array is all ones. This is `v(:, h)` in the Fortran code.
    single_model : bool, default=True
        If True, indicates that there is only one model. In this case,
        the model responsibilities are all 1, and the weighted
        responsibilities are equal to the mixture responsibilities. This
        unlocks an optimization to avoid unnecessary computation.

    Returns
    -------
    weighted_responsibilities : array, shape (n_samples, n_components, n_mixtures)
        Per-sample, per-component, per-mixture responsibilities weighted by
        the model responsibility. This is `u` in the Fortran code. Note that if
        `single_model` is True, weighted_responsibilities is a view of
        mixture_responsibilities (no copy is made).
    
    Notes
    -----
    Fortran reference:
        u(bstrt:bstp) = v(bstrt:bstp,h) * z(bstrt:bstp,i,j,h)
    """
    # Alias for clarity with Fortran code
    z = mixture_responsibilities
    v_h = model_responsibilities


    assert z.ndim == 3, f"z must be 3D, got {z.ndim}D"
    assert v_h.ndim == 1, f"v_h must be 1D, got {v_h.ndim}D"
    assert z.shape[0] == v_h.shape[0], (
        f"z.shape[0]={z.shape[0]} != v_h.shape[0]={v_h.shape[0]}"
    )
    # fast-path: for num_models == 1, v is all ones and thus u == z
    if single_model:
        return z  # NOTE: returns a view of z, no copy 
    else:
        # Weight mixture responsibilities by model responsibility
        u = v_h[:, None, None] * z  # shape: (n_samples, nw, num_mix)
    return u


def compute_source_scores(
        *,
        pdftype: int,
        y: SourceArray3D,
        rho: ParamsArray,
        comp_slice: slice,
):
    """Compute the score function (fp) to evaluate the non-Gaussianity of sources.

    Parameters
    ----------
    pdftype : int
        Probability density function type. Currently, only pdftype=0 (Gaussian) is supported.
    y : np.ndarray
        Scaled sources for the current model, of shape (n_samples, n_components, n_mixtures).
        Not modified.
    rho : np.ndarray
        Shape parameters of shape (n_components, n_mixtures). Not modified.
    comp_slice : slice
        Slice object containing component indices for the current model.
    Returns
    -------
    out_scores : np.ndarray
        The computed score function, of shape (n_samples, n_components, n_mixtures).
        This is out_scores modified in-place. out_scores == fp in Fortran code.
    """
    # Shape assertions for new dimension standard
    N1, nw, num_mix = y.shape
    assert y.shape == (N1, nw, num_mix), f"y shape {y.shape} != (N1, nw, num_mix)"
    assert rho.shape[0] >= nw, f"rho.shape[0]={rho.shape[0]} must be >= nw={nw}"
    assert rho.shape[1] == num_mix, f"rho.shape[1]={rho.shape[1]} != num_mix={num_mix}"
    assert comp_slice.stop - comp_slice.start == nw, f"len(comp_slice)={comp_slice.stop - comp_slice.start} != nw={nw}"

    # !--- get fp, zfp
    if pdftype == 0:
        #-------------------------------FORTRAN CODE-------------------------------------
        # if (rho(j,comp_list(i,h)) == dble(1.0)) then
        # fp(bstrt:bstp) = sign(dble(1.0),y(bstrt:bstp,i,j,h))
        # else if (rho(j,comp_list(i,h)) == dble(2.0)) then
        # fp(bstrt:bstp) = y(bstrt:bstp,i,j,h) * dble(2.0)
        # else
        # tmpvec(bstrt:bstp) = log(abs(y(bstrt:bstp,i,j,h)))
        # tmpvec2(bstrt:bstp) = exp((rho(j,comp_list(i,h))-dble(1.0))*tmpvec(bstrt:bstp))
        # fp(bstrt:bstp) = rho(j,comp_list(i,h)) * sign(dble(1.0),y(bstrt:bstp,i,j,h)) *
        #--------------------------------------------------------------------------------
        
        # Get components for this model
        rho_h = rho[comp_slice, :]

        # Masks: Laplacian (rho==1), Gaussian (rho==2); generalized Gaussian otherwise
        lap_mask = (torch.isclose(rho_h, torch.tensor(1.0), atol=1e-12))
        gau_mask = (torch.isclose(rho_h, torch.tensor(2.0), atol=1e-12))

        # Default: generalized Gaussian score function        
        # Step 1. Compute |y|^(rho_h - 1) in-place
        out_scores = torch.abs(y)                  # out_scores = |y|
        torch.log(out_scores, out=out_scores)         # log(|y|)
        torch.multiply(rho_h - 1.0, out_scores, out=out_scores)
        torch.exp(out_scores, out=out_scores)         # |y|^(rho_h - 1)

        # Step 2. Multiply by rho_h and sign(y) without np.sign allocation
        out_scores *= rho_h * torch.where(y >= 0, 1.0, -1.0)
        
        # Overwrite with Laplacian/Gaussian score function where needed
        if lap_mask.any():
            # FIXME: Use a small loop to avoid fancy boolean indexing allocation
            out_scores[:, lap_mask] = torch.sign(y[:, lap_mask], out=out_scores[:, lap_mask])
        if gau_mask.any():
            # FIXME: Use a small loop to avoid fancy boolean indexing allocation
            out_scores[:, gau_mask] = torch.multiply(y[:, gau_mask], 2.0, out=out_scores[:, gau_mask])
    elif pdftype == 2:
        raise NotImplementedError()
    elif pdftype == 3:
        raise NotImplementedError()
    elif pdftype == 4:
        raise NotImplementedError()
    elif pdftype == 1:
        raise NotImplementedError()
    else:
        raise ValueError(
            f"Invalid pdftype value: {pdftype}. "
            "Expected values are 0, 1, 2, 3, or 4."
        )
    return out_scores


def accumulate_scores(
        *,
        scores,
        responsibilities,
        scale_params,
        comp_slice: slice,
):
    """
    Accumulate per-sample, per-component mixture scores and ufp sufficient statistics from responsibilities and score functions.
    
    Parameters
    ----------
    scores : np.ndarray
        The score function (fp) of shape (n_samples, n_components, n_mixtures).
        Not modified.
    responsibilities : np.ndarray
        The responsibilities (u) of shape (n_samples, n_components, n_mixtures).
        Not modified.
    scale_params : np.ndarray
        Scale parameters (sbeta) of shape (n_components, n_mixtures).
        Not modified.
    
    Returns
    -------
    out_ufp : np.ndarray
        The score function weighted by model-weighted mixture-responsibilities.
        shape (n_samples, n_components, n_mixtures), modified in place.
    out_g : np.ndarray
        out_ufp further weighted by the per-component mixture scale parameters, summed
        over mixtures shape (n_samples, n_components), modified in place.
    """
    # Shape assertions for new dimension standard
    N1, nw, num_mix = scores.shape
    assert responsibilities.shape == scores.shape, f"responsibilities shape {responsibilities.shape} != scores shape {scores.shape}"
    assert scale_params.shape[0] >= nw, f"scale_params.shape[0]={scale_params.shape[0]} must be >= nw={nw}"
    assert scale_params.shape[1] == num_mix, f"scale_params.shape[1]={scale_params.shape[1]} != num_mix={num_mix}"
     
    u = responsibilities
    fp = scores
    sbeta = scale_params
    sbeta_h = sbeta[comp_slice, :] # components for this model
    #--------------------------FORTRAN CODE-------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # ufp(bstrt:bstp) = u(bstrt:bstp) * fp(bstrt:bstp)
    # ufp[bstrt-1:bstp] = u[bstrt-1:bstp] * fp[bstrt-1:bstp]
    # (bstrt:bstp,i) = g(bstrt:bstp,i) + sbeta(j,comp_list(i,h)) * ufp(bstrt:bstp)
    #---------------------------------------------------------------
    # === Subsection: Accumulate Statistics for Parameter accumulators ===
    # !--- get g
    # if update_A:

    ufp = torch.multiply(u, fp)
    # Same as torch.einsum('tnj,nj->tn', ufp, sbeta_h) but faster and we update g inplace
    g = torch.sum(ufp * sbeta_h, dim=-1)
    return ufp, g

def accumulate_c_stats(
        *,
        X: DataArray2D,
        model_responsibilities: SamplesVector,
        vsum: float,
        do_reject: bool = False,
        out_numer: ComponentsVector,
        out_denom: ComponentsVector,
        ) -> Tuple[ParamsModelArray, ParamsModelArray]:
    """
    Get sufficient statistics for model bias vector c.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_features, n_samples). The input data.
    model_responsibilities : np.ndarray
        Shape (n_samples,). The model responsibilities.
    vsum : float
        The total responsibility mass for the current model across all samples
        (i.e. the sum across samples).
    do_reject : bool, default False
        Currently only False (no outlier rejection) is supported.
    out_numer : np.ndarray
        Shape (n_components,). Accumulator for the numerator of the c
        gradient For one model. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components,). Accumulator for the denominator of the c
        gradient for one model. This array is mutated in-place and returned.
    
    Returns
    -------
    out_numer : np.ndarray
        Sum of X weighted by the model responsibilities (n_features, n_features).
        This is out_numer mutated in-place.
    out_denom : np.ndarray
        Total responsibility mass for the model, broadcasted to (n_components, n_mixtures)
        This is out_denom mutated in-place.
    
    Notes
    -----
    Fortran reference:
        tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,xstrt:xstp) )
        dc_numer_tmp(:,i,h) = dc_numer_tmp(:,i,h) + tmpsum
        dc_denom_tmp(:,i,h) = dc_denom_tmp(:,i,h) + vsum
    """
    # Alias for clarity with Fortran code
    dataseg = X
    v = model_responsibilities
    assert X.ndim == 2, f"X must be 2D, got {X.ndim}D"
    assert v.ndim == 1, f"model responsibilities must be 1D, got {v.ndim}D"
    assert X.shape[1] == v.shape[0], (
        f"X n_samples {X.shape[1]} != model responsibilities length {v.shape[0]}"
    )
    assert vsum.numel() == 1, f"vsum must be a scalar, got {vsum}"
    assert out_numer.shape == (X.shape[0],), (
        f"out_numer shape {out_numer.shape} != (n_components,) "
    )
    assert out_denom.shape == (X.shape[0],), (
        f"out_denom shape {out_denom.shape} != (n_components,) "
        f"= ({X.shape[0]},)"
    )
    if do_reject:
        raise NotImplementedError()
        # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,dataseg(seg)%goodinds(xstr
    else:
        #--------------------------FORTRAN CODE-------------------------
        # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,xstrt:xstp) )
        #---------------------------------------------------------------
        tmpsum_c_vec = dataseg @ v  # Shape: (n_components,)
    out_numer += tmpsum_c_vec
    out_denom += vsum
    return out_numer, out_denom


def accumulate_alpha_stats(
        *,
        usum: ParamsArray,
        vsum: torch.float64,
        out_numer: ParamsArray,
        out_denom: ParamsArray
        ) -> Tuple[ParamsArray, ParamsArray]:
    """
    Accumulate sufficient statistics for alpha (mixture weights) update.
    
    Parameters
    ----------
    usum : np.ndarray
        Shape (nw, num_mix). The per-source, per-mixture total
        responsibility mass across all samples (i.e. the sum across samples).
    vsum : float
        the total responsibility mass for the current model across all
        samples (i.e. the sum across samples).
    out_numer : np.ndarray
        Shape (nw, num_mix). Accumulator for the numerator of the alpha
        gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (nw, num_mix). Accumulator for the denominator of the alpha
        gradient. This array is mutated in-place and returned.
    
    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.
    
    Notes
    -----
    Fortran reference:
        for (h = num_models) ... for (i = 1, nw) ... for (j = 1, num_mix)
            dalpha_numer_tmp(j,comp_list(i,h)) = dalpha_numer_tmp(j,comp_list(i,h)) + usum
            dalpha_denom_tmp(j,comp_list(i,h)) = dalpha_denom_tmp(j,comp_list(i,h)) + vsum
    
    ..Warning::
        If you pass a slice created via fancy indexing, NumPy will return a copy, so you
        must reassign the result back into the parent array.
    """
    assert out_numer.shape == usum.shape, (
        f"out_numer shape {out_numer.shape} != usum shape {usum.shape}"
    )
    assert vsum.numel() == 1, f"vsum must be a scalar, got {vsum}"
    # -------------------------------FORTRAN--------------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # dalpha_numer_tmp(j,comp_list(i,h)) = dalpha_numer_tmp(j,comp_list(i,h)) + usum
    # dalpha_denom_tmp(j,comp_list(i,h)) = dalpha_denom_tmp(j,comp_list(i,h)) + vsum
    # -----------------------------------------------------------------------
    # TODO: if this is a single contribution per iteration, safe to assign directly.
    out_numer += usum
    out_denom += vsum
    return out_numer, out_denom


def accumulate_mu_stats(
        *,
        ufp: SourceArray3D,
        rho: ParamsArray,
        sbeta: ParamsArray,
        y: SourceArray3D,
        out_numer: ParamsArray,
        out_denom: ParamsArray,
        ) -> Tuple[ParamsArray, ParamsArray]:
    """
    Accumulate sufficient statistics for mu (location) update.
    
    Parameters
    ----------
    ufp : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The elementwise product of
        responsibilities and score function.
    rho : np.ndarray
        Shape (n_components, n_mixtures). The shape parameters.
    sbeta : np.ndarray
        Shape (n_components, n_mixtures). The scale parameters.
    y : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The scaled sources for the
        current model.
    out_numer : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the numerator of the
        mu gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the denominator of the
        mu gradient. This array is mutated in-place and returned.
    
    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.
    
    Notes
    -----
    Fortran reference:
        for (h = num_models) ... for (i = 1, nw) ... for (j = 1, num_mix)
            dmu_numer_tmp(j,comp_list(i,h)) = dmu_numer_tmp(j,comp_list(i,h)) + sum( ufp(bstrt:bstp) )
            dmu_denom_tmp(j,comp_list(i,h)) = dmu_denom_tmp(j,comp_list(i,h)) + rho(j,comp_list(i,h)) * sum( abs(ufp(bstrt:bstp)) )
    """
    assert out_numer.ndim == 2, f"out_numer must be 2D, got {out_numer.ndim}D" 
    assert out_denom.ndim == 2, f"out_denom must be 2D, got {out_denom.ndim}D"
    # -------------------------------FORTRAN--------------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # tmpsum = sum( ufp(bstrt:bstp) )
    # dmu_numer_tmp(j,comp_list(i,h)) = dmu_numer_tmp(j,comp_list(i,h)) + tmpsum
    # -----------------------------------------------------------------------
    tmpsum_mu = ufp.sum(dim=0)  # shape: (nw, num_mix)
    out_numer += tmpsum_mu
    # -------------------------------FORTRAN--------------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # if (rho(j,comp_list(i,h)) .le. dble(2.0)) then
    # tmpsum = sbeta(j,comp_list(i,h)) * sum( ufp(bstrt:bstp) / y(bstrt:bstp,i,j,h) )
    # dmu_denom_tmp(j,comp_list(i,h)) = dmu_denom_tmp(j,comp_list(i,h)) + tmpsum 
    # else
    # tmpsum = sbeta(j,comp_list(i,h)) * sum( ufp(bstrt:bstp) * fp(bstrt:bstp) )
    # -----------------------------------------------------------------------
    if torch.all(rho <= 2.0):
        mu_denom_sum = torch.sum(ufp / y, dim=0)
        tmpsum_mu_denom = (sbeta * mu_denom_sum)
        out_denom += tmpsum_mu_denom
    else:
        raise NotImplementedError("Generalized Gaussian mu update not implemented yet.")


def accumulate_beta_stats(
        *,
        usum: ParamsArray,
        rho: ParamsArray,
        ufp: SourceArray3D,
        y: SourceArray3D,
        out_numer: ParamsArray,
        out_denom: ParamsArray,
        ) -> Tuple[ParamsArray, ParamsArray]:
    """
    Get the numerator and denominator for the scale accumulators.

    Parameters
    ----------
    usum : np.ndarray
        Shape (n_components, n_mixtures). The per-source, per-mixture total
        responsibility mass across all samples (i.e. the sum across samples).
    rho : np.ndarray
        Shape (n_components, n_mixtures). The shape parameters.
    ufp : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The elementwise product of
        responsibilities and score function.
    y : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The scaled sources for the
        current model.
    out_numer : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the numerator of the
        beta gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the denominator of the
        beta gradient. This array is mutated in-place and returned.

    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.
    """
    # -------------------------------FORTRAN--------------------------------
    # dbeta_numer_tmp(j,comp_list(i,h)) = dbeta_numer_tmp(j,comp_list(i,h)) + usum
    # dbeta_numer_tmp[j - 1, comp_list[i - 1, h - 1] - 1] += usum
    # ----------------------------------------------------------------------
    out_numer += usum  # shape: (nw, num_mix)
    # -------------------------------FORTRAN--------------------------------
    # if (rho(j,comp_list(i,h)) .le. dble(2.0)) then
    # tmpsum = sum( ufp(bstrt:bstp) * y(bstrt:bstp,i,j,h) )
    # dbeta_denom_tmp(j,comp_list(i,h)) =  dbeta_denom_tmp(j,comp_list(i,h)) + tmpsum
    # ----------------------------------------------------------------------
    if torch.all(rho <= 2.0):
        # (s=samples, i=n_components, j=num_mixtures)
        # Same as torch.einsum("sij,sij->ij", ufp, y)
        out_denom += torch.sum(ufp * y, dim=0)  # shape: (nw, num_mix)
    else:
        raise NotImplementedError()


def accumulate_rho_stats(
        *,
        y: SourceArray3D,
        rho: ParamsArray,
        epsdble: float,
        u: SourceArray3D,
        usum: ParamsArray,
        out_numer: ParamsArray,
        out_denom: ParamsArray,
        ) -> Tuple[ParamsArray, ParamsArray]:
    """
    Compute the numerator and denominator for the shape accumulators.

    Parameters
    y : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The scaled sources for the
        current model.
    rho : np.ndarray
        Shape (n_components, n_mixtures). The shape parameters.
    epsdble : float
        Default 1.0e-16. Floor for log-exp underflow handling;
        values with exp(rho*log|y|) < epsdble zero the log contribution.
    u : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The responsibilities.
    usum : np.ndarray
        Shape (n_components, n_mixtures). The per-source, per-mixture total
        responsibility mass across all samples (i.e. the sum across samples).
    out_numer : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the numerator of the
        rho gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the denominator of the
        rho gradient. This array is mutated in-place and returned.
    
    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.
    """
    assert y.ndim == 3, f"y must be 3D, got {y.ndim}D"
    assert u.shape == y.shape, f"u shape {u.shape} != y shape {y.shape}"
    assert rho.ndim == 2, f"rho must be 2D, got {rho.ndim}D"
    assert usum.shape == rho.shape, f"usum shape {usum.shape} != rho shape {rho.shape}"
    assert np.isscalar(epsdble), f"epsdble must be a scalar, got {epsdble}"
    assert out_numer.shape == rho.shape, f"out_numer shape {out_numer.shape} != rho shape {rho.shape}"
    assert out_denom.shape == rho.shape, f"out_denom shape {out_denom.shape} != rho shape {rho.shape}"
    # -------------------------------FORTRAN--------------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # tmpy(bstrt:bstp) = abs(y(bstrt:bstp,i,j,h))
    # logab(bstrt:bstp) = log(tmpy(bstrt:bstp))
    # tmpy(bstrt:bstp) = exp(rho(j,comp_list(i,h))*logab(bstrt:bstp))
    # logab(bstrt:bstp) = log(tmpy(bstrt:bstp))
    # where (tmpy(bstrt:bstp) < epsdble)
            #    logab(bstrt:bstp) = dble(0.0)
            # end where
    # logab[bstrt-1:bstp][tmpy[bstrt-1:bstp] < epsdble] = 0.0
    # tmpsum = sum( u(bstrt:bstp) * tmpy(bstrt:bstp) * logab(bstrt:bstp) )
    # drho_numer_tmp(j,comp_list(i,h)) =  drho_numer_tmp(j,comp_list(i,h)) + tmpsum
    # drho_denom_tmp(j,comp_list(i,h)) =  drho_denom_tmp(j,comp_list(i,h)) + usum
    # ----------------------------------------------------------------------
    tmpy = torch.empty_like(y)
    logab = torch.empty_like(y)
    # 1. log|y| into logab, and also keep |y| in tmpy
    torch.abs(y, out=tmpy)
    torch.log(tmpy, out=logab)
    # 2. logab = rho * log|y|
    torch.multiply(rho[None, :, :], logab, out=logab)
    # 3. tmpy = |y|^rho
    torch.exp(logab, out=tmpy)
    # 4) Zero small contributions in logab based on tmpy threshold
    logab[tmpy < epsdble] = 0.0
    # 5. Numerator: sum(u * |y|^rho * log(|y|^rho)) over samples
    torch.multiply(u, tmpy, out=tmpy)
    torch.multiply(tmpy, logab, out=tmpy)
    # Sum over samples -> (nw, num_mix) and accumulate
    out_numer += tmpy.sum(dim=0)
    out_denom += usum
    if torch.any(rho > 2.0):
        raise NotImplementedError()
    return out_numer, out_denom


def accumulate_sigma2_stats(
        *,
        model_responsibilities: SamplesVector,
        source_estimates: SourceArray2D,
        vsum: float,
        out_numer: ComponentsVector,
        out_denom: ComponentsVector,
):
    """Get sufficient statistics for sigma2 (noise variance) update.

    Parameters
    ----------
    model_responsibilities : np.ndarray
        Shape (n_samples,). The model responsibilities. This is `v(:, h)` in the Fortran
        code.
    source_estimates : np.ndarray
        Shape (n_samples, n_components). The source estimates. This is `b` in the
        Fortran code.
    vsum : float
        The sum of the model responsibilities.
    out_numer : np.ndarray
        Shape (n_components,). The numerator accumulator.
    out_denom : np.ndarray
        Shape (n_components,). The denominator accumulator.
    
    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components,). This is out_numer
        mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components,). This is out_denom
        mutated in-place.
    
    Notes
    -----
    Fortran reference:
        for (h = num_models) ... for (i = 1, nw)
            tmpsum = sum( v(bstrt:bstp,h) * b(bstrt:bstp,i,h) * b(bstrt:bstp,i,h) )
            dsigma2_numer_tmp(i,h) = dsigma2_numer_tmp(i,h) + tmpsum
            dsigma2_denom_tmp(i,h) = dsigma2_denom_tmp(i,h) + vsum

    Fortran accumulators dsigma2_numer and dsigma2_denom in all iterations, but that is not
    necessary.
    """
    # Alias for clarity with Fortran code
    v_h = model_responsibilities
    b = source_estimates
    assert v_h.ndim == 1, f"v_h must be 1D, got {v_h.ndim}D"
    assert b.ndim == 2, f"b must be 2D, got {b.ndim}D"
    assert vsum.numel() == 1, f"vsum must be a scalar, got {vsum}"
    assert b.shape[0] == v_h.shape[0], f"samples dimension mismatch {b.shape[0]} != {v_h.shape[0]}"
    #--------------------------FORTRAN CODE-------------------------
    # !print *, myrank+1,':', thrdnum+1,': getting dsigma2 ...'; call flush(6)
    # tmpsum = sum( v(bstrt:bstp,h) * b(bstrt:bstp,i,h) * b(bstrt:bstp,i,h) )
    # dsigma2_numer_tmp(i,h) = dsigma2_numer_tmp(i,h) + tmpsum
    # dsigma2_denom_tmp(i,h) = dsigma2_denom_tmp(i,h) + vsum
    #---------------------------------------------------------------
    # weighted column-wise sum of squares: (s=n_samples, i=n_components)
    # Same as torch.einsum('s,si,si->i', v_h, b, b)
    out_numer += v_h @ (b**2)
    out_denom += vsum
    return out_numer, out_denom


def accumulate_kappa_stats(
        *,
        ufp: SourceArray3D,
        fp: SourceArray3D,
        sbeta: ParamsArray,
        usum: ParamsArray,
        out_numer: ParamsArray,
        out_denom: ParamsArray,
        ) -> Tuple[ParamsArray, ParamsArray]:
    """
    Get sufficient statistics for kappa (curvature) update.
    
    Parameters
    ufp : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The elementwise product of
        responsibilities and score function.
    fp : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The score function.
    sbeta : np.ndarray
        Shape (n_components, n_mixtures). The scale parameters.
    usum : np.ndarray
        Shape (n_components, n_mixtures). The per-source, per-mixture total
        responsibility mass across all samples (i.e. the sum across samples).
    out_numer : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the numerator of the
        kappa gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the denominator of the
        kappa gradient. This array is mutated in-place and returned.

    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.
    
    Notes
    -----
    Fortran reference:
        for (h = num_models) ... for (i = 1, nw) ... for (j = 1, num_mix)
            tmpsum = sum( ufp(bstrt:bstp) * fp(bstrt:bstp) )
            dkappa_numer_tmp(j,comp_list(i,h)) = dkappa_numer_tmp(j,comp_list(i,h)) + sbeta(j,comp_list(i,h)) * tmpsum
            dkappa_denom_tmp(j,comp_list(i,h)) = dkappa_denom_tmp(j,comp_list(i,h)) + usum
    """
    assert ufp.ndim == 3, f"ufp must be 3D, got {ufp.ndim}D"
    assert fp.ndim == 3, f"fp must be 3D, got {fp.ndim}D"
    assert sbeta.ndim == 2, f"sbeta must be 2D, got {sbeta.ndim}D"
    assert out_numer.ndim == 2, f"out_numer must be 2D, got {out_numer.ndim}D"
    assert out_denom.ndim == 2, f"out_denom must be 2D, got {out_denom.ndim}D"
    assert ufp.shape == fp.shape, f"ufp {ufp.shape} != fp {fp.shape}"
    assert out_numer.shape == sbeta.shape and out_denom.shape == sbeta.shape
    #--------------------------FORTRAN CODE-------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # tmpsum = sum( ufp(bstrt:bstp) * fp(bstrt:bstp) ) * sbeta(j,comp_list(i,h))**2
    # dkappa_numer_tmp(j,i,h) = dkappa_numer_tmp(j,i,h) + tmpsum
    # dkappa_denom_tmp(j,i,h) = dkappa_denom_tmp(j,i,h) + usum
    #---------------------------------------------------------------
    # (s=n_samples, i=n_components, j=n_mixtures)
    # Same as torch.einsum('sij,sij->ij', ufp, fp) * (sbeta**2)
    out_numer += (ufp * fp).sum(dim=0) * (sbeta**2)
    out_denom += usum
    return out_numer, out_denom


def accumulate_lambda_stats(
        *,
        fp: SourceArray3D,
        y: SourceArray3D,
        u: SourceArray3D,
        usum: ParamsArray,
        out_numer: ParamsArray,
        out_denom: ParamsArray,
) -> Tuple[ParamsArray, ParamsArray]:
    """
    Get sufficient statistics for lambda (nonlinearity) update.

    Parameters
    fp : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The score function.
    y : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The scaled sources for the
        current model.
    u : np.ndarray
        Shape (n_samples, n_components, n_mixtures). The mixture responsibilities
        weighted by the model responsibility.
    usum : np.ndarray
        Shape (n_components, n_mixtures). The per-source, per-mixture total
        responsibility mass across all samples (i.e. the sum across samples).
    out_numer : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the numerator of the
        lambda gradient. This array is mutated in-place and returned.
    out_denom : np.ndarray
        Shape (n_components, n_mixtures). Accumulator for the denominator of the
        lambda gradient. This array is mutated in-place and returned.

    Returns
    -------
    out_numer : np.ndarray
        Updated numerator accumulator (n_components, n_mixtures). This is
        out_numer mutated in-place.
    out_denom : np.ndarray
        Updated denominator accumulator (n_components, n_mixtures). This is
        out_denom mutated in-place.

    Notes
    -----
    Fortran reference:
        for (h = num_models) ... for (i = 1, nw) ... for (j = 1, num_mix)
            tmpsum = sum( fp(bstrt:bstp) * y(bstrt:bstp,i,j,h) )
            dlambda_numer_tmp(j,comp_list(i,h)) = dlambda_numer_tmp(j,comp_list(i,h)) + tmpsum
            dlambda_denom_tmp(j,comp_list(i,h)) = dlambda_denom_tmp(j,comp_list(i,h)) + usum
    """
    assert fp.ndim == 3, f"fp must be 3D, got {fp.ndim}D"
    assert y.ndim == 3, f"y must be 3D, got {y.ndim}D"
    assert u.ndim == 3, f"u must be 3D, got {u.ndim}D"
    assert out_numer.ndim == 2, f"out_numer must be 2D, got {out_numer.ndim}D"
    assert out_denom.ndim == 2, f"out_denom must be 2D, got {out_denom.ndim}D"
    assert fp.shape == y.shape == u.shape, f"fp {fp.shape}, y {y.shape}, u {u.shape} shape mismatch"
    assert out_numer.shape == out_denom.shape == usum.shape, (
        f"out_numer {out_numer.shape}, out_denom {out_denom.shape}, usum {usum.shape} shape mismatch"
    )
    # ---------------------------FORTRAN CODE---------------------------
    # tmpvec(bstrt:bstp) = fp(bstrt:bstp) * y(bstrt:bstp,i,j,h) - dble(1.0)
    # tmpsum = sum( u(bstrt:bstp) * tmpvec(bstrt:bstp) * tmpvec(bstrt:bstp) )
    # dlambda_numer_tmp(j,i,h) = dlambda_numer_tmp(j,i,h) + tmpsum
    # dlambda_denom_tmp(j,i,h) = dlambda_denom_tmp(j,i,h) + usum
    # ------------------------------------------------------------------
    # (s=n_samples, i=n_components, j=n_mixtures)
    # Same as (u * (fp * y - 1.0)**2).sum(dim=0) but avoids 3 intermediate allocations
    tmp = fp * y # one allocation
    tmp -= 1.0
    tmp **= 2
    # Same as torch.einsum('sij,sij->ij', u, tmp)
    out_numer += (u * tmp).sum(dim=0)
    out_denom += usum
    return out_numer, out_denom


def accum_updates_and_likelihood(
        *,
        config,
        accumulators,
        state,
        total_LL,  # this is LLtmp in Fortran
        iter
        ):
    # !--- add to the cumulative dtmps
    # ...
    #--------------------------FORTRAN CODE-------------------------
    # call MPI_REDUCE(dgm_numer_tmp,dgm_numer,num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dalpha_numer_tmp,dalpha_numer,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dalpha_denom_tmp,dalpha_denom,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dmu_numer_tmp,dmu_numer,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dmu_denom_tmp,dmu_denom,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dbeta_numer_tmp,dbeta_numer,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dbeta_denom_tmp,dbeta_denom,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(drho_numer_tmp,drho_numer,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(drho_denom_tmp,drho_denom,num_mix*num_comps,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dc_numer_tmp,dc_numer,nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    # call MPI_REDUCE(dc_denom_tmp,dc_denom,nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    #---------------------------------------------------------------
    num_comps = config.n_components
    nw = num_comps
    num_models = config.n_models
    do_reject = config.do_reject
    do_newton = config.do_newton
    newt_start = config.newt_start

    A = state.A
    gm = state.gm
    mu = state.mu
    
    dgm_numer = accumulators.dgm_numer
    dA = accumulators.dA
    dAK = accumulators.dAK

    if do_newton:
        dbaralpha_numer = accumulators.newton.dbaralpha_numer
        dbaralpha_denom = accumulators.newton.dbaralpha_denom
        dsigma2_numer = accumulators.newton.dsigma2_numer
        dsigma2_denom = accumulators.newton.dsigma2_denom
        dkappa_numer = accumulators.newton.dkappa_numer
        dkappa_denom = accumulators.newton.dkappa_denom
        dlambda_numer = accumulators.newton.dlambda_numer
        dlambda_denom = accumulators.newton.dlambda_denom
    # if update_A:
    # call MPI_REDUCE(dWtmp,dA,nw*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    
    assert dA.shape == (32, 32, 1) == (nw, nw, num_models)
    Wtmp_working = torch.zeros((num_comps, num_comps))

    if do_newton and iter >= newt_start:
        #--------------------------FORTRAN CODE-------------------------
        # call MPI_REDUCE(dbaralpha_numer_tmp,dbaralpha_numer,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dbaralpha_denom_tmp,dbaralpha_denom,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dkappa_numer_tmp,dkappa_numer,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dkappa_denom_tmp,dkappa_denom,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dlambda_numer_tmp,dlambda_numer,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dlambda_denom_tmp,dlambda_denom,num_mix*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dsigma2_numer_tmp,dsigma2_numer,nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        # call MPI_REDUCE(dsigma2_denom_tmp,dsigma2_denom,nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
        #---------------------------------------------------------------
        assert dbaralpha_denom[0, 0, 0] == 30504


    # if (seg_rank == 0) then
    # if update_A:
    if do_newton and iter >= newt_start:
        #--------------------------FORTRAN CODE-------------------------
        # baralpha = dbaralpha_numer / dbaralpha_denom
        # sigma2 = dsigma2_numer / dsigma2_denom
        # kappa = dble(0.0)
        # lambda = dble(0.0)
        #---------------------------------------------------------------
        # shape (num_mix, num_comps, num_models)
        baralpha = dbaralpha_numer / dbaralpha_denom
        # shape (num_comps, num_models)
        sigma2 = dsigma2_numer / dsigma2_denom
        kappa = torch.zeros((num_comps, num_models), dtype=torch.float64)
        lambda_ = torch.zeros((num_comps, num_models), dtype=torch.float64)
        for h, _ in enumerate(range(num_models), start=1):
            comp_slice = get_component_slice(h, num_comps)
            h_idx = h - 1  # For easier indexing
            # NOTE: VECTORIZED
            # In Fortran, this is a nested for loop...
            # for do (h = 1, num_models)
            # for do j = 1, num_mix

            # These 6 variables don't exist in Fortran.
            baralpha_h = baralpha[:, :, h_idx]
            dkappa_numer_h = dkappa_numer[:, :, h_idx]
            dkappa_denom_h = dkappa_denom[:, :, h_idx]
            dlambda_numer_h = dlambda_numer[:, :, h_idx]
            dlambda_denom_h = dlambda_denom[:, :, h_idx]
            # Get the component indices for this model 'h'

            # Calculate dkap for all mixtures 
            # dkap = dkappa_numer(j,i,h) / dkappa_denom(j,i,h)
            # kappa(i,h) = kappa(i,h) + baralpha(j,i,h) * dkap
            dkap = dkappa_numer_h / dkappa_denom_h
            # --- Update kappa ---
            # Calculate all update terms and sum along the mixture axis
            kappa_update = torch.sum(baralpha_h * dkap, dim=1)
            kappa[:, h_idx] += kappa_update

            # --- Update lambda_ ---
            #--------------------------FORTRAN CODE-------------------------
            # lambda(i,h) = lambda(i,h) + ...
            #       baralpha(j,i,h) * ( dlambda_numer(j,i,h)/dlambda_denom(j,i,h) + dkap * mu(j,comp_list(i,h))**2 )
            #---------------------------------------------------------------
            # mu_selected will have shape (num_mix, nw)
            mu_selected = mu[comp_slice, :]

            # Calculate the full lambda update term
            lambda_inner_term = (dlambda_numer_h / dlambda_denom_h) + (dkap * mu_selected**2)
            lambda_update = torch.sum(baralpha_h * lambda_inner_term, dim=1)
            lambda_[:, h_idx] += lambda_update
            # end do (j)
            # end do (i)
        # end do (h)
        # if (print_debug) then
    # end if (do_newton .and. iter >= newt_start)
    elif not do_newton and iter >= newt_start:
        raise NotImplementedError()  # pragma no cover 

    no_newt = False
    for h, _ in enumerate(range(num_models), start=1):
        comp_slice = get_component_slice(h, num_comps)
        h_index = h - 1
        #--------------------------FORTRAN CODE-------------------------
        # if (print_debug) then
        # print *, 'dA ', h, ' = '; call flush(6)
        # call DSCAL(nw*nw,dble(-1.0)/dgm_numer(h),dA(:,:,h),1)
        # dA(i,i,h) = dA(i,i,h) + dble(1.0)
        #---------------------------------------------------------------
        if do_reject:
            raise NotImplementedError()
        else:
            dA[:, :, h - 1] *= -1.0 / dgm_numer[h - 1]
        
        # basically the same as np.fill_diagonal where fill value is diag + 1.0
        diag = dA[:, :, h_index].diagonal()
        idx = torch.arange(nw)
        dA[idx, idx, h_index] = diag + 1.0        
        # if (print_debug) then

        global posdef
        posdef = True
        if do_newton and iter >= newt_start:
            # in Fortran, this is a nested loop..
            #--------------------------FORTRAN CODE-------------------------
            # do i = 1,nw ... do k = 1,nw
            # if (i == k) then
            # Wtmp(i,i) = dA(i,i,h) / lambda(i,h)
            # else
            # sk1 = sigma2(i,h) * kappa(k,h)
            # sk2 = sigma2(k,h) * kappa(i,h)
            #---------------------------------------------------------------
            # We have to do a numpy -> torch -> numpy round trip here because
            # torch.fill_diagonal only accepts scalar fill values.
            # on-diagonal elements
            fill_values = dA[:, :, h - 1].diagonal().numpy() / lambda_[:, h - 1].numpy()
            np.fill_diagonal(Wtmp_working.numpy(), fill_values)
            # off-diagonal elements
            i_indices, k_indices = np.meshgrid(np.arange(nw), np.arange(num_comps), indexing='ij')
            off_diag_mask = i_indices != k_indices
            sk1 = sigma2[i_indices, h-1].numpy() * kappa[k_indices, h-1].numpy()
            sk2 = sigma2[k_indices, h-1].numpy() * kappa[i_indices, h-1].numpy()
            positive_mask = (sk1 * sk2 > 0.0)
            if np.any(~positive_mask):
                posdef = False
                no_newt = True
                # This is a placeholder to see if this condition is hit
                assert 1 == 0
            condition_mask = positive_mask & off_diag_mask
            if np.any(condition_mask):
                # # Wtmp(i,k) = (sk1*dA(i,k,h) - dA(k,i,h)) / (sk1*sk2 - dble(1.0))
                numerator = sk1 * dA.numpy()[i_indices, k_indices, h-1] - dA.numpy()[k_indices, i_indices, h-1]
                denominator = sk1 * sk2 - 1.0
                Wtmp_working.numpy()[condition_mask] = (numerator / denominator)[condition_mask]
            # end if (i == k)
            # end do (k)
            # end do (i)
        # end if (do_newton .and. iter >= newt_start)
        elif not do_newton and iter >= newt_start:
            raise NotImplementedError()  # pragma no cover
        if ((not do_newton) or (not posdef) or (iter < newt_start)):
            #  Wtmp = dA(:,:,h)
            assert Wtmp_working.shape == dA[:, :, h - 1].squeeze().shape == (nw, nw)
            Wtmp_working = (dA[:, :, h - 1].squeeze()).clone()
            assert Wtmp_working.shape == (32, 32) == (nw, nw)
        
        #--------------------------FORTRAN CODE-------------------------
        # call DSCAL(nw*nw,dble(0.0),dA(:,:,h),1)
        # call DGEMM('N','N',nw,nw,nw,dble(1.0),A(:,comp_list(:,h)),nw,Wtmp,nw,dble(1.0),dA(:,:,h),nw) 
        #---------------------------------------------------------------
        dA[:, :, h - 1] = 0.0
        dA[:, :, h - 1] += torch.matmul(A[:, comp_slice], Wtmp_working)
    # end do (h)

    zeta = torch.zeros(num_comps, dtype=torch.float64)
    for h, _ in enumerate(range(num_models), start=1):
        comp_slice = get_component_slice(h, num_comps)
        h_index = h - 1
        # NOTE: I had an indexing bug in the looped version of this code.
        # But it didn't seem to affect the results.
        
        #--------------------------FORTRAN CODE-------------------------
        # dAk(:,comp_list(i,h)) = dAk(:,comp_list(i,h)) + gm(h)*dA(:,i,h)
        # zeta(comp_list(i,h)) = zeta(comp_list(i,h)) + gm(h)
        #---------------------------------------------------------------
        source_columns = gm[h - 1] * dA[:, :, h - 1]
        dAK[comp_slice, :] += source_columns
        zeta[comp_slice] += gm[h - 1]
    
    #--------------------------FORTRAN CODE-------------------------
    # dAk(:,k) = dAk(:,k) / zeta(k)
    # nd(iter,:) = sum(dAk*dAk,1)
    # ndtmpsum = sqrt(sum(nd(iter,:),mask=comp_used) / (nw*count(comp_used)))
    #---------------------------------------------------------------
    dAK[:,:] /= zeta  # Broadcasting division
    # nd is (num_iters, num_comps) in Fortran, but we only store current iteration
    nd = torch.sum(dAK * dAK, dim=0)  # Python-only variable name
    assert nd.shape == (num_comps,)

    # comp_used should be 32 length vector of True
    # In Fortran Comp used was always be an all True bollean representation of comp_slice
    # Unless identify_shared_comps was run. I have no plans to implement that.
    comp_used = torch.ones(num_comps, dtype=bool)
    assert isinstance(comp_used, torch.Tensor)
    assert comp_used.shape == (num_comps,)
    assert comp_used.dtype == torch.bool
    ndtmpsum = torch.sqrt(torch.sum(nd) / (nw * torch.count_nonzero(comp_used)))
    # end if (update_A)
    
    # if (seg_rank == 0) then
    if do_reject:
        raise NotImplementedError()
        # LL(iter) = LLtmp2 / dble(numgoodsum*nw)
    else:
        # LL(iter) = LLtmp2 / dble(all_blks*nw)
        # XXX: In the Fortran code LLtmp2 is the summed LLtmps across processes.
        likelihood = total_LL / (all_blks * nw)
    # TODO: figure out what needs to be returned here (i.e. it is defined in thic func but rest of the program needs it)
    return (likelihood, ndtmpsum, no_newt)


def update_params(
        *,
        config,
        state,
        accumulators,
        metrics,
        wc,
):
    n_models = config.n_models
    nw = config.n_components
    do_newton = config.do_newton
    newt_start = config.newt_start
    newtrate = config.newtrate
    newt_ramp = config.newt_ramp
    do_reject = config.do_reject
    lrate0 = config.lrate
    rholrate0 = config.rholrate

    W = state.W
    A = state.A
    c = state.c
    alpha = state.alpha
    mu = state.mu
    sbeta = state.sbeta
    rho = state.rho
    gm = state.gm

    dgm_numer = accumulators.dgm_numer
    dalpha_numer = accumulators.dalpha_numer
    dalpha_denom = accumulators.dalpha_denom
    dc_numer = accumulators.dc_numer
    dc_denom = accumulators.dc_denom
    dmu_numer = accumulators.dmu_numer
    dmu_denom = accumulators.dmu_denom
    dbeta_numer = accumulators.dbeta_numer
    dbeta_denom = accumulators.dbeta_denom
    drho_numer = accumulators.drho_numer
    drho_denom = accumulators.drho_denom
    dAK = accumulators.dAK

    iter = metrics.iter
    lrate = metrics.lrate
    rholrate = metrics.rholrate


    num_models = n_models
    # if (seg_rank == 0) then
    # if update_gm:
    if do_reject:
        raise NotImplementedError()
        # gm = dgm_numer / dble(numgoodsum)
    else:
        gm[:] = dgm_numer / all_blks 
    # end if (update_gm)

    # if update_alpha:
    # assert alpha.shape == (num_comps, num_mix)
    alpha[:, :] = dalpha_numer / dalpha_denom

    # if update_c:
    # assert c.shape == (nw, num_models)
    c[:, :] = dc_numer / dc_denom
    
    # === Section: Apply Parameter accumulators & Rescale ===
    # Apply accumulated statistics to update parameters, then rescale and refresh W/wc.
    # !print *, 'updating A ...'; call flush(6)
    # global lrate, rholrate, lrate0, rholrate0, newtrate, newt_ramp
    if (iter < share_start or (iter % share_iter > 5)):
        if do_newton and (not metrics.no_newt) and (iter >= newt_start):
            # lrate = min( newtrate, lrate + min(dble(1.0)/dble(newt_ramp),lrate) )
            # rholrate = rholrate0
            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            lrate = min(newtrate, lrate + min(1.0 / newt_ramp, lrate))
            rholrate = rholrate0
            A[:, :] -= lrate * dAK
        else:
            if not posdef:
                print("Hessian not positive definite, using natural gradient")
                assert 1 == 0
            
            lrate = min(
                lrate0, lrate + min(1 / newt_ramp, lrate)
                )
            
            rholrate = rholrate0
            

            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            A[:, :] -= lrate * dAK
            
        # end if do_newton
    # end if (update_A)

    # if update_mu:
    mu[:, :] += dmu_numer / dmu_denom
     
    # if update_beta:
    
    sbeta[:, :] *= torch.sqrt(dbeta_numer / dbeta_denom)
    sbetatmp[:, :] = torch.minimum(torch.tensor(invsigmax), sbeta)

    sbeta[:, :] = torch.maximum(torch.tensor(invsigmin), sbetatmp)  # Fill?

    # end if (update_beta)

    rho[:, :] += (
            rholrate
            * (
                1.0
                - (rho / torch.special.psi(1.0 + 1.0 / rho))
            * drho_numer
            / drho_denom
        )
    )
    rhotmp = torch.minimum(torch.tensor(maxrho), rho) # shape (num_comps, num_mix)
    assert rhotmp.shape == (config.n_components, config.n_mixtures)
    rho[:, :] = torch.maximum(torch.tensor(minrho), rhotmp)

    # !--- rescale
    # !print *, 'rescaling A ...'; call flush(6)
    # from seed import A_FORTRAN
    if doscaling:
        # calculate the L2 norm for each column of A and then use it to normalize that
        # column and scale the corresponding columns in mu and sbeta, but only if the
        # norm is positive.
        # NOTE: this shadows a global variable Anrmk
        Anrmk = torch.linalg.norm(A, dim=0)
        positive_mask = Anrmk > 0
        if positive_mask.all():
            A[:, positive_mask] /= Anrmk[positive_mask]
            mu[positive_mask, :] *= Anrmk[positive_mask, None]
            sbeta[positive_mask, :] /= Anrmk[positive_mask, None]
        else:
            raise NotImplementedError()            
    # end if (doscaling)

    if (share_comps and (iter >= share_start) and (iter-share_iter % share_iter == 0)):
        raise NotImplementedError()
    else:
        global free_pass
        free_pass = False
    
    W[:, :, :], wc[:, :] = get_unmixing_matrices(
        iterating=True,
        c=c,
        A=A,
        comp_slice=get_component_slice(1, nw), # FIXME
        W=W,
        num_models=num_models,
    )


    
    # if (print_debug) then
    
    # call MPI_BCAST(gm,num_models,MPI_DOUBLE_PRECISION,0,seg_comm,ierr)
    # ...
    return lrate, rholrate, state, wc



if __name__ == "__main__":
    seed_array = 12345 # + myrank. For reproducibility
    np.random.seed(seed_array)
    rng = np.random.default_rng(seed_array)
    # newtrate = 1.0  # default is 0.5 but config file sets it to 1.0
    # do_reject = False
    # lrate = 0.05 # default of program is 0.1 but config file set it to 0.05

    # lrate0 = 0.05 # this is set to the user-passed lrate value in the Fortran code
    # rholrate = 0.05
    # rholrate0 = 0.05


    # load_sphere = False
    # do_sphere = True

    # !-------------------- GET THE DATA ------------------------
    fpath = Path("/Users/scotterik/devel/projects/amica-python/amica/eeglab_data.set")
    raw = mne.io.read_raw_eeglab(fpath)
    blocks_in_sample, num_samples, all_blks = get_seg_list(raw)
    
    dataseg: np.ndarray = raw.get_data() # shape (n_channels, n_times) = (32, 30504)
    dataseg *= 1e6  # Convert to microvolts
    # Check our value against the Fortran output

    #if (do_reject) then
    #    allocate(dataseg(1)%gooddata(dataseg(1)%lastdim),stat=ierr); call tststat(ierr)
    #    allocate(dataseg(1)%goodinds(dataseg(1)%lastdim),stat=ierr); call tststat(ierr)
    #    dataseg(1)%gooddata = .true.
    #    dataseg(1)%numgood = dataseg(1)%lastdim
    #    do j = 1,dataseg(1)%numgood
    #       dataseg(1)%goodinds(j) = j
    #    end do
    # end if


    S, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL = amica(
        X=dataseg,
        max_iter=200,
        tol=1e-7,
        lrate=0.05,
        rholrate=0.05,
        newtrate=1.0,
        )
    # call write_output
    # The final comparison with Fortran saved outputs.
    # If we set tol to .0001 then we can assert that Amica solves at iteration 106
    # Just like Fortran does.
    LL_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/LL")
    assert_almost_equal(LL, LL_f, decimal=4)
    assert_allclose(LL, LL_f, atol=1e-4)

    A_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/A")
    A_f = A_f.reshape((32, 32), order="F")
    assert_almost_equal(A, A_f, decimal=2)

    alpha_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/alpha")
    alpha_f = alpha_f.reshape((3, 32), order="F")
    # Remember that alpha (and sbeta, mu etc) are (num_comps, num_mix) in Python
    assert_almost_equal(alpha, alpha_f.T, decimal=2)

    c_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/c")
    c_f = c_f.reshape((32, 1), order="F")
    assert_almost_equal(c, c_f)


    comp_list_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/comp_list", dtype=np.int32)
    # Something weird is happening there. I expect (num_comps, num_models) = (32, 1)
    comp_list_f = np.reshape(comp_list_f, (32, 2), order="F")


    gm_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/gm")
    assert gm == gm_f == np.array([1.])

    mean_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/mean")
    assert_almost_equal(mean, mean_f)

    mu_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/mu", dtype=np.float64)
    mu_f = mu_f.reshape((3, 32), order="F")
    assert_almost_equal(mu, mu_f.T, decimal=0)

    rho_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/rho", dtype=np.float64)
    rho_f = rho_f.reshape((3, 32), order="F")
    assert_almost_equal(rho, rho_f.T, decimal=2)

    S_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/S", dtype=np.float64)
    S_f = S_f.reshape((32, 32,), order="F")
    assert_almost_equal(S, S_f)

    sbeta_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/sbeta", dtype=np.float64)
    sbeta_f = sbeta_f.reshape((3, 32), order="F")
    assert_almost_equal(sbeta, sbeta_f.T, decimal=1)

    W_f = np.fromfile("/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug/W", dtype=np.float64)
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
      X : ndarray, shape (n_channels, n_times)
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
      X_centered = X - mean[:, np.newaxis]

      # 2. Apply sphering
      X_sphered = S @ X_centered

      # 3. Apply ICA unmixing (this is the key step)
      sources = W[:, :, 0] @ X_sphered  # For single model, use W[:,:,0]

      return sources

sources_python = get_amica_sources(
    dataseg, W, S, mean
)
sources_fortran = get_amica_sources(
    dataseg, W_f, S_f, mean_f
)
# Now lets check the correlation between the two sources
# Taking a subset to avoid memory issues
corrs = np.zeros(sources_python.shape[0])
for i in range(sources_python.shape[0]):
    corr = np.corrcoef(
        sources_python[i, ::10],
        sources_fortran[i, ::10]
    )[0, 1]
    corrs[i] = corr
assert np.all(np.abs(corr) > 0.99)  # Should be very high correlation

info = mne.create_info(
    ch_names=[f"IC{i}" for i in range(sources_python.shape[0])],
    sfreq=raw.info['sfreq'],
    ch_types='eeg'
)

raw_src_python = mne.io.RawArray(sources_python, info)
raw_src_fortran = mne.io.RawArray(sources_fortran, info)

mne.viz.set_browser_backend("matplotlib")
fig = raw_src_python.plot(scalings=dict(eeg=.3))
fig.savefig("/Users/scotterik/devel/projects/amica-python/figs/amica_sources_python.png")
plt.close(fig)
fig = raw_src_fortran.plot(scalings=dict(eeg=.3))
fig.savefig("/Users/scotterik/devel/projects/amica-python/figs/amica_sources_fortran.png")
plt.close(fig)
