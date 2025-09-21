from copy import copy
from pathlib import Path
import time
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import torch

from amica._batching import BatchLoader, choose_batch_size
from amica.constants import (
    mineig,
    minlog,
    epsdble,
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
    minlrate,
    lratefact,
    rholratefact,
)

from amica.state import (
    AmicaConfig,
    AmicaState,
    IterationMetrics,
    get_initial_state,
    initialize_accumulators,
)
from amica._types import (
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

from amica.kernels import (
    compute_preactivations,
    compute_source_densities,
    compute_model_loglikelihood_per_sample,
    compute_mixture_responsibilities,
    compute_total_loglikelihood_per_sample,
    compute_model_responsibilities,
    compute_weighted_responsibilities,
    compute_source_scores,
    accumulate_scores,
    accumulate_c_stats,
    accumulate_alpha_stats,
    accumulate_mu_stats,
    accumulate_beta_stats,
    accumulate_rho_stats,
    accumulate_sigma2_stats,
    accumulate_kappa_stats,
    accumulate_lambda_stats,
)

from amica.linalg import (
    get_unmixing_matrices,
    compute_sign_log_determinant,
    get_initial_model_log_likelihood,
    pre_whiten,
)

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
        initial_weights=None,
        initial_scales=None,
        initial_locations=None,
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
        `X.shape[0]` to process all samples at once.", but note that this may lead to
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
    initial_weights : array-like, shape (n_components, n_components), optional
        Initial weights for the mixture components. If None, weights are initialized randomly.
        This is meant to be used for testing and debugging purposes only.
    initial_scales : array-like, shape (n_components, n_mixtures), optional
        Initial scales (sbeta) for the mixture components. If None, scales are
        initialized randomly. This is meant to be used for testing and debugging purposes
        only.
    initial_locations : array-like, shape (n_components, n_mixtures), optional
        Initial locations (mu) for the mixture components. If None, locations are
        initialized randomly. This is meant to be used for testing and debugging purposes
        only.
    """
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
    if config.do_reject:
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
        initial_weights=initial_weights,
        initial_scales=initial_scales,
        initial_locations=initial_locations,
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


def _core_amica(
        X,
        *,
        config,
        state,
        sldet,
        initial_weights=None,
        initial_scales=None,
        initial_locations=None,
):
    """Runs the AMICA algorithm.
    
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
        initialized randomly. This is meant to be used for testing and debugging purposes
        only.
    initial_locations : array-like, shape (n_components, n_mixtures), optional
        Initial locations (mu) for the mixture components. If None, locations are
        initialized randomly. This is meant to be used for testing and debugging purposes
        only.
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
    if initial_locations is None:
        raise NotImplementedError("Random initialization of mu not yet implemented")
    else:
        assert initial_locations.shape == (num_comps, num_mix)
        initial_locations = torch.as_tensor(initial_locations, dtype=torch.float64)
    state.mu = state.mu + 0.05 * (1.0 - 2.0 * initial_locations)
    # load_beta:
    if initial_scales is None:
        raise NotImplementedError("Random initialization of sbeta not yet implemented")
    else:
        assert initial_scales.shape == (num_comps, num_mix)
        initial_scales = torch.as_tensor(initial_scales, dtype=torch.float64)
    state.sbeta = 1.0 + 0.1 * (0.5 - initial_scales)
    # load_c:
    state.c.fill_(0.0)
    
    # load_A:
    for h, _ in enumerate(range(num_models), start=1):
        h_index = h - 1
        comp_slice = get_component_slice(h=h, n_components=num_comps)
        if initial_weights is None:
            raise NotImplementedError("Random initialization of weights not yet implemented")
        else:
            assert initial_weights.shape == (num_comps, num_comps)
            initial_weights = torch.as_tensor(initial_weights, dtype=torch.float64)
        state.A[:, comp_slice] = 0.01 * (0.5 - initial_weights)
        idx = torch.arange(num_comps)
        cols = h_index * num_comps + idx
        state.A[idx, cols] = 1.0
        Anrmk = torch.linalg.norm(state.A[:, cols], dim=0)
        state.A[:, cols] /= Anrmk   
    # end load_A
    
    W, wc = get_unmixing_matrices(
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
    

def optimize(
        *,
        X: DataTensor2D,
        sldet: float,
        wc: ParamsModelTensor,
        config: AmicaConfig,
        state: AmicaState,
):
    """Main optimization loop for AMICA."""
    # Just set all convergence creterion to the user specific tol
    min_dll = config.tol
    min_nd = config.tol
    
    # These variables can be updated in the loop
    leave = False
    iteration = 1
    do_newton = config.do_newton
    numincs = 0  # number of consecutive iterations where LL increased by less than tol
    lrate = config.lrate
    rholrate = config.rholrate

    # Initialize accumulators container
    accumulators = initialize_accumulators(config)
    # We allocate these separately.
    Dsum = torch.zeros(config.n_models, dtype=torch.float64)
    Dsign = torch.zeros(config.n_models, dtype=torch.float64)
    loglik = torch.zeros((X.shape[0],), dtype=torch.float64)  # per sample log likelihood
    LL = torch.zeros(max(1, config.max_iter), dtype=torch.float64)  # likelihood history

    c_start = time.time()
    c1 = time.time()
    while iteration <= config.max_iter:
        accumulators.reset()
        loglik.fill_(0.0)
        metrics = IterationMetrics(
            iter=iteration,
            lrate=lrate,
            rholrate=rholrate,
        )

        # !----- get determinants
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

        if config.do_reject:
            raise NotImplementedError()
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
        batch_loader = BatchLoader(X, axis=0, batch_size=config.batch_size)
        for batch_idx, (data_batch, batch_indices) in enumerate(batch_loader):
            for h, _ in enumerate(range(config.n_models), start=1):
                comp_slice = get_component_slice(h, config.n_components)
                h_index = h - 1
                
                # =======================================================================
                #                       Expectation Step (E-step)
                # =======================================================================

                # 1. --- Compute source pre-activations
                # !--- get b
                b = compute_preactivations(
                    X=data_batch,
                    unmixing_matrix=state.W[:, :, h_index],
                    bias=wc[:, h_index],
                    do_reject=config.do_reject,
                )
                
                # 2. --- Source densities, and per-sample mixture log-densities (logits) ---
                y, z = compute_source_densities(
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
                    size=(data_batch.shape[0], config.n_models),
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

            if config.do_reject:
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

                # NOTE: u is a view of z, so changes to u will affect z (and vice versa)
                u = compute_weighted_responsibilities(
                    mixture_responsibilities=z,
                    model_responsibilities=v_h,
                    single_model=(config.n_models == 1),
                )
                z = None; del z # guard against use of stale name. u owns that memory now
                usum = u.sum(dim=0)  # shape: (nw, num_mix)

                fp = compute_source_scores(
                    pdftype=config.pdftype,
                    y=y,
                    rho=state.rho,
                    comp_slice=comp_slice,
                )

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
                if do_newton and iteration >= config.newt_start:
                    if iteration == 50 and batch_indices.start == 0:
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
                # end if (do_newton and iteration >= newt_start)

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
        
        # End of these lifetimes
        del b, fp, g, u, ufp, usum, vsum, v, v_h, y

        likelihood, ndtmpsum = accum_updates_and_likelihood(
            X=X,
            config=config,
            accumulators=accumulators,
            state=state,
            total_LL=loglik.sum(),
            iteration=iteration
        )
        metrics.loglik = likelihood
        metrics.ndtmpsum = ndtmpsum
        # return accumulators, metrics

        # ==============================================================================
        ndtmpsum = metrics.ndtmpsum
        LL[iteration - 1] = metrics.loglik
        # init
        numdecs = 0
             
        # !----- display log likelihood of data
        # if (seg_rank == 0) then
        c2 = time.time()
        t0 = c2 - c1
        #  if (mod(iter,outstep) == 0) then

        if (iteration % outstep) == 0:
            print(
                f"Iteration {iteration}, lrate = {lrate:.3f}, LL = {LL[iteration - 1]:.3f}, "
                f"nd = {ndtmpsum:.3f}, D = {Dsum.max():.3f} {Dsum.min():.3f} "
                f"took {t0:.2f} seconds"
                )
            c1 = time.time()

        # !----- check whether likelihood is increasing
        # if (seg_rank == 0) then
        # ! if we get a NaN early, try to reinitialize and startover a few times 
        if torch.isnan(LL[iteration - 1]):
            raise RuntimeError(f"Log Likelihood is NaN at iteration {iteration}")
        # end if
        if iteration == 2:
            assert not torch.isnan(LL[iteration - 1])
            assert not (LL[iteration - 1] < LL[iteration - 2])
        if iteration > 1:
            if (LL[iteration - 1] < LL[iteration - 2]):
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
                        if iteration == 2:
                            assert 1 == 0
                        if iteration > config.newt_start:
                            raise NotImplementedError()
                            rholrate0 *= rholratefact
                        if config.do_newton and iteration > config.newt_start:
                            print("Reducing maximum Newton lrate")
                            newtrate *= lratefact
                            assert 1 == 0 # stop to check that value
                        numdecs = 0
                    # end if (numdecs >= maxdecs)
                # end if (lrate vs minlrate)
            # end if LL
            if use_min_dll:
                if (LL[iteration - 1] - LL[iteration - 2]) < min_dll:
                    numincs += 1
                    if numincs > maxincs:
                        leave = True
                        print(
                            f"Exiting because likelihood increasing by less than {min_dll} "
                            f"for more than {maxincs} iterations ..."
                            )
                else:
                    numincs = 0
                if iteration == 2:
                    assert numincs == 0
            else:
                raise NotImplementedError() # pragma no cover
            if use_grad_norm:
                if ndtmpsum < min_nd:
                    leave = True
                    print(
                        f"Exiting because norm of weight gradient less than {min_nd:.6f}"
                    )
        # end if (iter > 1)
        if config.do_newton and (iteration == config.newt_start):
            print("Starting Newton ... setting numdecs to 0")
            numdecs = 0
        # call MPI_BCAST(leave,1,MPI_LOGICAL,0,seg_comm,ierr)
        # call MPI_BCAST(startover,1,MPI_LOGICAL,0,seg_comm,ierr)
        if leave:
            return state, LL
        # else:
        # !----- do accumulators: gm, alpha, mu, sbeta, rho, W
        # the updated lrate & rholrate for the next iteration
        lrate, rholrate, state, wc = update_params(
            X=X,
            config=config,
            state=state,
            accumulators=accumulators,
            metrics=metrics,
            wc=wc,
        )
        # !----- reject data
        if config.do_reject:
            raise NotImplementedError()
        iteration += 1
        # end if/else
    # end while
    c_end = time.time()
    print(f"Finished in {c_end - c_start:.2f} seconds")
    return state, LL


def accum_updates_and_likelihood(
        *,
        X,
        config,
        accumulators,
        state,
        total_LL,  # this is LLtmp in Fortran
        iteration
        ):
    # !--- add to the cumulative dtmps
    # ...
    #--------------------------FORTRAN CODE-------------------------
    # call MPI_REDUCE(dgm_numer_tmp,dgm_numer,num_models,MPI_DOUBLE_PRECISION,MPI_S...
    # ...
    # if update_A:
    # call MPI_REDUCE(dWtmp,dA,nw*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_co...
    nw = config.n_components
    Wtmp_working = torch.zeros((config.n_components, config.n_components))
    # if (seg_rank == 0) then
    if config.do_newton and iteration >= config.newt_start:
        #--------------------------FORTRAN CODE-------------------------
        # baralpha = dbaralpha_numer / dbaralpha_denom
        # sigma2 = dsigma2_numer / dsigma2_denom
        # kappa = dble(0.0)
        # lambda = dble(0.0)
        #---------------------------------------------------------------
        # shape (num_mix, num_comps, num_models)
        baralpha = accumulators.newton.dbaralpha_numer / accumulators.newton.dbaralpha_denom
        sigma2 = accumulators.newton.dsigma2_numer / accumulators.newton.dsigma2_denom
        kappa = torch.zeros((config.n_components, config.n_models), dtype=config.dtype)
        lambda_ = torch.zeros((config.n_components, config.n_models), dtype=config.dtype)
        
        for h, _ in enumerate(range(config.n_models), start=1):
            comp_slice = get_component_slice(h, config.n_components)
            h_idx = h - 1

            # These 6 variables don't exist in Fortran.
            baralpha_h = baralpha[:, :, h_idx]
            dkappa_numer_h = accumulators.newton.dkappa_numer[:, :, h_idx]
            dkappa_denom_h = accumulators.newton.dkappa_denom[:, :, h_idx]
            dlambda_numer_h = accumulators.newton.dlambda_numer[:, :, h_idx]
            dlambda_denom_h = accumulators.newton.dlambda_denom[:, :, h_idx]

            # Calculate dkap for all mixtures 
            # dkap = dkappa_numer(j,i,h) / dkappa_denom(j,i,h)
            # kappa(i,h) = kappa(i,h) + baralpha(j,i,h) * dkap
            # --- Update kappa ---
            dkap = dkappa_numer_h / dkappa_denom_h
            kappa[:, h_idx] += torch.sum(baralpha_h * dkap, dim=1)

            # --- Update lambda_ ---
            #--------------------------FORTRAN CODE-------------------------
            # lambda(i,h) = lambda(i,h) + ...
            #       baralpha(j,i,h) * ( dlambda_numer(j,i,h)/dlambda_denom(j,i,h) + ...
            #---------------------------------------------------------------
            mu_selected = state.mu[comp_slice, :]
            # Calculate the full lambda update term
            lambda_inner_term = (dlambda_numer_h / dlambda_denom_h) + (dkap * mu_selected**2)
            lambda_update = torch.sum(baralpha_h * lambda_inner_term, dim=1)
            lambda_[:, h_idx] += lambda_update
            # end do (j)
            # end do (i)
        # end do (h)
        # if (print_debug) then
    # end if (do_newton .and. iter >= newt_start)

    for h, _ in enumerate(range(config.n_models), start=1):
        comp_slice = get_component_slice(h, config.n_components)
        h_index = h - 1
        #--------------------------FORTRAN CODE-------------------------
        # if (print_debug) then
        # print *, 'dA ', h, ' = '; call flush(6)
        # call DSCAL(nw*nw,dble(-1.0)/dgm_numer(h),dA(:,:,h),1)
        # dA(i,i,h) = dA(i,i,h) + dble(1.0)
        #---------------------------------------------------------------
        if config.do_reject:
            raise NotImplementedError()
        else:
            accumulators.dA[:, :, h - 1] *= -1.0 / accumulators.dgm_numer[h - 1]
        
        # basically the same as np.fill_diagonal where fill value is diag + 1.0
        diag = accumulators.dA[:, :, h_index].diagonal()
        idx = torch.arange(nw)
        accumulators.dA[idx, idx, h_index] = diag + 1.0
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
            diag = accumulators.dA[:, :, h - 1].diagonal()
            fill_values = diag / lambda_[:, h - 1]
            idx = torch.arange(Wtmp_working.shape[0])
            Wtmp_working[idx, idx] = fill_values

            # off-diagonal elements
            i_indices, k_indices = torch.meshgrid(
                torch.arange(config.n_components),
                torch.arange(config.n_components), indexing='ij'
                )
            off_diag_mask = i_indices != k_indices
            sk1 = sigma2[i_indices, h-1] * kappa[k_indices, h-1]
            sk2 = sigma2[k_indices, h-1] * kappa[i_indices, h-1]
            positive_mask = (sk1 * sk2 > 0.0)
            if torch.any(~positive_mask):
                raise RuntimeError(
                    f"Non-positive definite Hessian encountered in Newton update. "
                    f"Iteration {iteration} model {h}. Please try setting do_newton to False."
                    )
            condition_mask = positive_mask & off_diag_mask
            if torch.any(condition_mask):
                # # Wtmp(i,k) = (sk1*dA(i,k,h) - dA(k,i,h)) / (sk1*sk2 - dble(1.0))
                numerator = sk1 * accumulators.dA[i_indices, k_indices, h-1] - accumulators.dA[k_indices, i_indices, h-1]
                denominator = sk1 * sk2 - 1.0
                Wtmp_working[condition_mask] = (numerator / denominator)[condition_mask]
            # end if (i == k)
            # end do (k)
            # end do (i)
        # end if (do_newton .and. iter >= newt_start)
        if ((not config.do_newton) or (iteration < config.newt_start)):
            #  Wtmp = dA(:,:,h)
            assert Wtmp_working.shape == accumulators.dA[:, :, h - 1].squeeze().shape == (nw, nw)
            Wtmp_working = (accumulators.dA[:, :, h - 1].squeeze()).clone()
            assert Wtmp_working.shape == (32, 32) == (nw, nw)
        #--------------------------FORTRAN CODE-------------------------
        # call DSCAL(nw*nw,dble(0.0),dA(:,:,h),1)
        # call DGEMM('N','N',nw,nw,nw,dble(1.0),A(:,comp_list(:,h)),nw,Wtmp,nw,dble(1.0),dA(:,:,h),nw) 
        #---------------------------------------------------------------
        accumulators.dA[:, :, h - 1] = 0.0
        accumulators.dA[:, :, h - 1] += torch.matmul(state.A[:, comp_slice], Wtmp_working)
    # end do (h)

    zeta = torch.zeros(config.n_components, dtype=config.dtype)
    for h, _ in enumerate(range(config.n_models), start=1):
        comp_slice = get_component_slice(h, config.n_components)
        h_index = h - 1        
        #--------------------------FORTRAN CODE-------------------------
        # dAk(:,comp_list(i,h)) = dAk(:,comp_list(i,h)) + gm(h)*dA(:,i,h)
        # zeta(comp_list(i,h)) = zeta(comp_list(i,h)) + gm(h)
        #---------------------------------------------------------------
        source_columns = state.gm[h - 1] * accumulators.dA[:, :, h - 1]
        accumulators.dAK[comp_slice, :] += source_columns
        zeta[comp_slice] += state.gm[h - 1]
    
    #--------------------------FORTRAN CODE-------------------------
    # dAk(:,k) = dAk(:,k) / zeta(k)
    # nd(iter,:) = sum(dAk*dAk,1)
    # ndtmpsum = sqrt(sum(nd(iter,:),mask=comp_used) / (nw*count(comp_used)))
    #---------------------------------------------------------------
    accumulators.dAK[:,:] /= zeta  # Broadcasting division
    # nd is (num_iters, num_comps) in Fortran, but we only store current iteration
    nd = torch.sum(accumulators.dAK * accumulators.dAK, dim=0)  # Python-only variable name
    assert nd.shape == (config.n_components,)

    # comp_used should be a vector of True
    # In Fortran Comp used was always an all True boolean representation of comp_slice
    # Unless identify_shared_comps was run. I have no plans to implement that.
    comp_used = torch.ones(config.n_components, dtype=bool)
    assert isinstance(comp_used, torch.Tensor)
    assert comp_used.shape == (config.n_components,)
    assert comp_used.dtype == torch.bool
    ndtmpsum = torch.sqrt(torch.sum(nd) / (nw * torch.count_nonzero(comp_used)))
    # end if (update_A)
    
    # if (seg_rank == 0) then
    if config.do_reject:
        raise NotImplementedError()
    else:
        # LL(iter) = LLtmp2 / dble(all_blks*nw)
        # XXX: In the Fortran code LLtmp2 is the summed LLtmps across processes.
        likelihood = total_LL / (X.shape[0] * nw)
    return (likelihood, ndtmpsum)


def update_params(
        *,
        X,
        config,
        state,
        accumulators,
        metrics,
        wc,
):
    nw = config.n_components
    lrate0 = config.lrate
    rholrate0 = config.rholrate
    lrate = metrics.lrate
    rholrate = metrics.rholrate


    # if (seg_rank == 0) then
    # if update_gm:
    if config.do_reject:
        raise NotImplementedError()
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
    state.c[:, :] = accumulators.dc_numer / accumulators.dc_denom
    if torch.any(~torch.isfinite(state.c)):
        raise RuntimeError("Non-finite c encountered during update.")

    # === Section: Apply Parameter accumulators & Rescale ===
    # Apply accumulated statistics to update parameters, then rescale and refresh W/wc.
    # !print *, 'updating A ...'; call flush(6)
    if (metrics.iter < share_start or (metrics.iter % share_iter > 5)):
        if config.do_newton and (metrics.iter >= config.newt_start):
            # lrate = min( newtrate, lrate + min(dble(1.0)/dble(newt_ramp),lrate) )
            # rholrate = rholrate0
            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            lrate = min(config.newtrate, lrate + min(1.0 / config.newt_ramp, lrate))
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
    sbetatmp = torch.minimum(torch.tensor(invsigmax), state.sbeta)
    state.sbeta = torch.maximum(torch.tensor(invsigmin), sbetatmp)
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
    rhotmp = torch.minimum(torch.tensor(maxrho), state.rho) # shape (num_comps, num_mix)
    assert rhotmp.shape == (config.n_components, config.n_mixtures)
    state.rho = torch.maximum(torch.tensor(minrho), rhotmp)

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
            raise NotImplementedError()            
    # end if (doscaling)

    if share_comps:
        raise NotImplementedError()
    
    state.W, wc = get_unmixing_matrices(
        c=state.c,
        A=state.A,
        comp_slice=get_component_slice(1, nw), # FIXME
        W=state.W,
        num_models=config.n_models,
    )
    # if (print_debug) then
    # call MPI_BCAST(gm,num_models,MPI_DOUBLE_PRECISION,0,seg_comm,ierr)
    # ...
    return lrate, rholrate, state, wc



if __name__ == "__main__":
    main()


def main():
    seed_array = 12345 # + myrank. For reproducibility
    np.random.seed(seed_array)
    rng = np.random.default_rng(seed_array)

    # !-------------------- GET THE DATA ------------------------
    fpath = Path("/Users/scotterik/devel/projects/amica-python/amica/eeglab_data.set")
    raw = mne.io.read_raw_eeglab(fpath)

    dataseg: np.ndarray = raw.get_data().T # shape (n_times, n_channels) = (30504, 32)
    dataseg *= 1e6  # Convert to microvolts
    # Check our value against the Fortran output

    initial_weights = np.fromfile(
        "/Users/scotterik/devel/projects/amica-python/amica/amicaout_test/Wtmp.bin",
        dtype=np.float64
        )
    initial_weights = initial_weights.reshape((32, 32), order="F")
    initial_scales = np.fromfile(
        "/Users/scotterik/devel/projects/amica-python/amica/amicaout_test/sbetatmp.bin",
        dtype=np.float64
        )
    initial_scales = initial_scales.reshape((3, 32), order="F")
    initial_scales = initial_scales.T  # Match Our dimension standard
    initial_locations = np.fromfile(
        "/Users/scotterik/devel/projects/amica-python/amica/amicaout_test/mutmp.bin",
        dtype=np.float64
        )
    initial_locations = initial_locations.reshape((3, 32), order="F")
    initial_locations = initial_locations.T  # Match Our dimension standard
    S, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL = amica(
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
    # call write_output
    # The final comparison with Fortran saved outputs.
    # If we set tol to .0001 then we can assert that Amica solves at iteration 106
    # Just like Fortran does.

    amica_outdir = "/Users/scotterik/devel/projects/amica-python/amica/amicaout_debug"

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