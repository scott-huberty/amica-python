from copy import copy
from pathlib import Path
import time

import matplotlib.pyplot as plt
import mne
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose

# from pyAMICA.pyAMICA.amica_utils import psifun
from scipy import linalg
from scipy.special import gammaln, psi, softmax

from constants import (
    fix_init,
    mineig,
    dorho,
    rho0,
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
    min_dll,
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
    load_gm,
    load_A,
    load_mu,
    load_sbeta,
    load_beta,
    load_rho,
    load_c,
    load_alpha,
    do_opt_block,
    do_approx_sphere,
    LOG_2,
    LOG_SQRT_PI,
)

from seed import MUTMP, SBETATMP as sbetatmp, WTMP
from funmod import psifun

from state import (
    AmicaConfig,
    IterationMetrics,
    get_initial_state,
    initialize_updates,
    AmicaWorkspace,
)


import line_profiler

from typing import Annotated, Optional, Tuple, TypeAlias, Literal
import numpy.typing as npt

Samples1D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples,)"]
"""Alias for a 1D array with shape (n_samples,)."""

Data2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_features, n_samples)"]
"""Alias for a 2D array with shape (n_features, n_samples)."""

Weights2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_components, n_features)"]
"""Alias for a 2D array with shape (n_components, n_features)."""

Bias1D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_components,)"]
"""Alias for a 1D array with shape (n_components,)."""

Sources2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_components)"]
"""Alias for a 2D array with shape (n_samples, n_components)."""

Sources3D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_components, n_mixtures)"]
"""Alias for a 3D array with shape (n_samples, n_components, n_mixtures)."""

Params2D: TypeAlias  = Annotated[npt.NDArray[np.float64], "(n_components, n_mixtures)"]
"""Alias for a 2D array with shape (n_components, n_mixtures)."""

Index1D: TypeAlias  = Annotated[npt.NDArray[np.integer], "(n_features,)"]
"""Alias for a 1D array with shape (n_features,)."""

Buffer3D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_components, n_mixtures)"]
"""Alias for a 3D array with shape (n_samples, n_components, n_mixtures)."""

Likelihood2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_models)"]
"""Alias for a 2D array with shape (n_samples, n_models)."""

sbetatmp = sbetatmp.T


def get_component_indices(num_comps: int, num_models: int):
    """Return column-shaped comp_list and its boolean mask.

    Parameters
    - num_comps: number of components (nw)
    - num_models: number of models

    Returns
    - comp_list: ndarray of shape (num_comps, num_models), dtype=int64
    - comp_used: ndarray of shape (num_comps,), dtype=bool
    """
    comp_list = np.empty((num_comps, num_models), dtype=int)
    for h, _ in enumerate(range(num_models), start=1):
        h_index = h - 1
        col = h_index * num_comps + np.arange(1, num_comps + 1)
        comp_list[:, h_index] = col
    # Right now comp_used are always all True.
    # This will change if we implement shared components across models.
    comp_used = np.ones(num_comps, dtype=bool)  # Mask for used components
    return comp_list, comp_used


# TODO's
# - consider keeping z and z0 separate once chunking is implemented

def amica(
        X,
        *,
        whiten=True,
        centering=True,
        n_components=None,
        n_models=1,
        n_mixtures=3,
        pdftype=0,
        max_iter=500,
        tol=1e-7,
        lrate=0.05,
        rholrate=0.05,
        do_newton=True,
        newt_start=50,
        newtrate=1,
        newt_ramp=10,
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
    # Step 1: Create config and state objects (new dataclass approach)
    config = AmicaConfig(
        n_features=X.shape[0],  # Number of channels (corrected from X.shape[1])
        n_components=n_components if n_components is not None else X.shape[0],
        n_models=n_models,
        n_mixtures=n_mixtures,
        max_iter=max_iter,
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
    state = get_initial_state(config)
    
    # random_state = check_random_state(random_state)

    # Init
    if n_models > 1:
        raise NotImplementedError("n_models > 1 not yet supported")
    if do_reject:
        raise NotImplementedError("Sample rejection by log likelihood is not yet supported yet")
    dataseg = X
    do_mean = True if centering else False
    do_sphere = True if whiten else False
    # !---------------------------- get the mean --------------------------------
    nx = dataseg.shape[0]  # Number of channels
    # TODO: n_components gets set twice. Thats an anti-pattern.
    if n_components is None:
        n_components = nx
    print("getting the mean ...")
    meantmp = dataseg.sum(axis=1) # Sum across time points for each channel
    assert_almost_equal(meantmp[0], -113139.76889015333)
    mean = meantmp / dataseg.shape[1]  # Divide by number of time points to get mean
    assert_almost_equal(mean[0], -3.7090141912586323)
    assert_almost_equal(mean[1], -6.7516788952437894)
    assert_almost_equal(mean[2], 2.5880450259870105)
    # We did in two steps what we could have done in one:
    np.testing.assert_array_almost_equal(mean, dataseg.mean(axis=1))

    # !--- subtract the mean
    dataseg -= mean[:, np.newaxis]  # Subtract mean from each channel
    assert_almost_equal(dataseg[0, 0], -32.088471160303868)
    assert_almost_equal(dataseg[1, 0], 9.0595228188125141)
    assert_almost_equal(dataseg[2, 0], -29.36477079502998)

    # !------------------------ sphere the data -------------------------------
    Stmp = np.zeros((nx, nx))
    Stmp_2 = np.zeros((nx, nx))
    print(" Getting the covariance matrix ...")
    # call DSCAL(nx*nx,dble(0.0),Stmp,1)
    # Compute the covariance matrix
    # The Fortran code processes the data in blocks
    # and appears to only update the lower triangular part of the covariance matrix
    # But in practice, you usually want to compute the full covariance matrix?
    # e.g. Stmp = np.cov(dataseg)

    # call DSYRK('L','N',nx,blk_size(seg),dble(1.0),dataseg(seg)%data(:,bstrt:bstp),nx,dble(1.0),Stmp,nx)
    X = dataseg.copy()
    full_cov = X @ X.T
    Stmp_2[np.tril_indices(nx)] = full_cov[np.tril_indices(nx)]
   
    S = Stmp_2.copy()  # Copy the lower triangular part to S
    np.testing.assert_almost_equal(S[0, 0], 45778661.956294745, decimal=6)
    cnt = 30504 # Number of time points, as per the Fortran code
    # Normalize the covariance matrix by dividing by the number of samples
    # S is the final covariance matrix
    # call DSCAL(nx*nx,dble(1.0)/dble(cnt),S,1) 
    S /= cnt
    np.testing.assert_almost_equal(S[0, 0], 1500.7429175286763, decimal=6)

    # Better approach: vectorized.
    full_cov = dataseg @ dataseg.T
    # The fortran code only computes the lower triangle for efficiency
    # So lets set the upper triangle to zero for consistency    
    Stmp[np.tril_indices(nx)] = full_cov[np.tril_indices(nx)]
    np.testing.assert_almost_equal(Stmp, Stmp_2, decimal=7)

    #### Do Eigenvalue Decomposition ####
    lwork = 10 * nx * nx  # Work array size, as per Fortran code
    eigs = np.zeros(nx)
    eigv = np.zeros(nx)
    print(f"doing eig nx = {nx}, lwork = {lwork}")
    # call DCOPY(nx*nx,S,1,Stmp,1)
    Stmp = S.copy()  # Copy S to Stmp
    assert np.isclose(S[0, 0], 1500.7429175286763, atol=1e-6)
    assert np.isclose(S[0, 0], 1500.7429175286763, atol=1e-6)

    # call DSYEV('V','L',nx,Stmp,nx,eigs,work,lwork,info)
    eigs, eigv = np.linalg.eigh(Stmp)  # Eigenvalue decomposition
    assert eigs.ndim == 1
    assert len(eigs) == 32
    assert eigv.shape == (32, 32) # in Fortran, eigv is 32 and is used to store the reversed eigenvalues
    # eigv is == to Stmp in the Fortran code
    np.testing.assert_almost_equal(abs(eigv[0][0]), 0.01141531781264421, decimal=7)
    np.testing.assert_almost_equal(abs(eigv[0][1]), 0.022133957340276893, decimal=7)
    np.testing.assert_almost_equal(abs(eigv[1][0]), abs(-0.00048653972579690302), decimal=7)
    np.testing.assert_almost_equal(eigs[0], 4.8799005132501803, decimal=7)
    np.testing.assert_almost_equal(eigs[1], 6.9201197127079803, decimal=7)
    np.testing.assert_almost_equal(eigs[2], 7.6562147928880702, decimal=7)
    lowest_eigs = np.sort(eigs)[:3]
    want_low_eigs = [4.8799005132501803, 6.9201197127079803, 7.6562147928880702]
    biggest_eigs = np.sort(eigs)[-3:][::-1]
    want_big_eigs = [9711.1430838537090, 3039.6850435125002, 1244.4129447052057]
    np.testing.assert_array_almost_equal(
        lowest_eigs,
        want_low_eigs
    )
    np.testing.assert_array_almost_equal(
        biggest_eigs,
        want_big_eigs        
    )
    print(f"minimum eigenvalues: {lowest_eigs}")
    print(f"maximum eigenvalues: {biggest_eigs}")
    
    # wrappers for the fortran program either set pcakeep to nchans
    # or to nchans-1 in case of an average reference.
    pcakeep = n_components
    assert isinstance(pcakeep, int)
    # TODO: use np.linalg.matrix_rank?
    numeigs = min(pcakeep, sum(eigs > mineig))
    assert numeigs == nx == 32
    print(f"num eigs kept: {numeigs}")

    # if load_sphere:
    #    raise NotImplementedError()
    # else:
    # !--- get the sphering matrix
    print("Getting the sphering matrix ...")
    # reverse the order of eigenvectors
    Stmp2 = eigv.copy()[:, ::-1]  # Reverse the order of eigenvectors
    np.testing.assert_almost_equal(Stmp2[0, 0], eigv[0, 31], decimal=7)
    eigv = np.sort(eigs)[::-1]
    np.testing.assert_almost_equal(eigv[0], 9711.1430838537090, decimal=7)
    np.testing.assert_almost_equal(eigs[31], 9711.1430838537090, decimal=7) 
    np.testing.assert_almost_equal(abs(Stmp2[0, 0]), 0.21635948345763786, decimal=7)

    # do sphere
    if do_sphere:
        # This is a duplicate of the previous step
        print(f"doing eig nx = {nx}, lwork = {lwork}")
        assert S.shape == (nx, nx) == (32, 32)
        np.testing.assert_almost_equal(S[0, 0], 1500.7429175286763, decimal=6)
        Stmp = S.copy() # call DCOPY(nx*nx,S,1,Stmp,1)

        eigs, eigvecs = np.linalg.eigh(Stmp)  # eigvecs: columns are eigenvectors
        Stmp = eigvecs.copy()  # Overwrite Stmp with eigenvectors, just like Fortran

        min_eigs = eigs[:min(nx//2, 3)]
        max_eigs = eigs[::-1][:3] # eigs[nx:(nx-min(nx//2, 3)):-1]
        print(f"minimum eigenvalues: {min_eigs}")
        print(f"maximum eigenvalues: {max_eigs}")

        min_eigs_fortran = [4.8799005132501803, 6.9201197127079803, 7.6562147928880702]
        max_eigs_fortran = [9711.1430838537090, 3039.6850435125002, 1244.4129447052057]
        np.testing.assert_array_almost_equal(
            min_eigs,
            min_eigs_fortran
        )
        np.testing.assert_array_almost_equal(
            max_eigs,
            max_eigs_fortran
        )

        numeigs = min(pcakeep, sum(eigs > mineig))
        print(f"num eigs kept: {numeigs}")
        assert numeigs == nx == 32

        Stmp2 = np.zeros((numeigs, nx))
        assert Stmp2.shape == (numeigs, nx) == (32, 32)
        eigv = np.zeros(nx)
        assert eigv.shape == (nx,) == (32,)
        eigv = np.flip(eigs) # Reverse the eigenvalues
        Stmp2 = np.flip(Stmp[:, :nx], axis=1).T  # Reverse the order of eigenvectors (columns)
        np.testing.assert_almost_equal(abs(Stmp2[0, 0]), 0.21635948345763786)
        np.testing.assert_almost_equal(abs(Stmp2[0, 1]), 0.054216688971114729)
        np.testing.assert_almost_equal(abs(Stmp2[1, 0]), 0.43483598508694776)

        Stmp = Stmp2.copy()  # Copy the reversed eigenvectors to Stmp
        sqrt_eigv = np.sqrt(eigv).reshape(-1, 1)
        Stmp2 /= sqrt_eigv
        non_finite_check = ~np.isfinite(Stmp2)
        if non_finite_check.any():
            non_finite_indices = np.where(non_finite_check)[0]
            unique_rows_with_non_finite = np.unique(non_finite_indices)
            for i in unique_rows_with_non_finite:
                print(f"Non-finite value detected! i = {i}, eigv = {eigv[i]}")
            raise NotImplementedError("Non-finite values detected in Stmp2 after division.")
        
        sldet = 0.0 # Log determinant of the whitening matrix, initialized to zero
        sldet -= 0.5 * np.sum(np.log(eigv))
        np.testing.assert_almost_equal(sldet, -65.935050239880198, decimal=7)
        np.testing.assert_almost_equal(abs(Stmp2[0, 0]), 0.0021955369949589743, decimal=7)

        if numeigs == nx:
            # call DSCAL(nx*nx,dble(0.0),S,1) 
            if do_approx_sphere:
                raise NotImplementedError()
            else:
                # call DCOPY(nx*nx,Stmp2,1,S,1)  
                S = Stmp.T @ Stmp2
        else:
            # if (do_approx_sphere) then
            raise NotImplementedError()
        np.testing.assert_almost_equal(S[0, 0], 0.043377346952119616, decimal=7)
    else:
        # !--- just normalize by the channel variances (don't sphere)
        raise NotImplementedError()
   
    print("Sphering the data...")

    # Apply the sphering matrix to the data (whitening)
    fieldsize = dataseg.shape[1]
    assert fieldsize == 30504
    # TODO: this is all very inefficient. We should be able to do this in one go with a single matrix multiplication
    # e.g. 
    
    # -------------------- FORTRAN CODE ---------------------------------------
    # call DSCAL(nx*blk_size(seg),dble(0.0),xtmp(:,1:blk_size(seg)),1)
    # call DGEMM('N','N',nx,blk_size(seg),nx,dble(1.0),S,nx,dataseg(seg)%data(:,bstrt:bstp),nx,dble(1.0),xtmp(:,1:blk_size(seg)),nx)
    # call DCOPY(nx*blk_size(seg),xtmp(:,1:blk_size(seg)),1,dataseg(seg)%data(:,bstrt:bstp),1)
    # -------------------------------------------------------------------------
    X = dataseg.copy() # TODO: unnecessary copy?
    xtmp = np.zeros(shape=((nx, dataseg.shape[1])))
    xtmp[:, :] = S @ X # Apply the sphering matrix
    dataseg[:, :] = xtmp[:, :]

    # Lets check dataseg
    assert_almost_equal(dataseg[0, 0], -0.18746213684159407, decimal=7)
    assert_almost_equal(dataseg[0, 1], -0.15889933957961194, decimal=7)
    assert_almost_equal(dataseg[1, 0], 0.44527165614822528, decimal=7)
    assert_almost_equal(dataseg[-1, -1], -0.79980176796607527, decimal=7)
    assert_almost_equal(dataseg[0, 15252], 0.78073780455880826, decimal=7)
    assert_almost_equal(dataseg[0, 29696], 0.31586289746943713, decimal=7)
    assert_almost_equal(dataseg[0, 29700], 0.34534557250740822, decimal=7)
    assert_almost_equal(dataseg[15, 29701], 1.2248470873789368, decimal=7)
    assert_almost_equal(dataseg[31, 0],0.70942796672956288, decimal=7)
    assert_almost_equal(dataseg[20, 29710], 0.26516668950937816, decimal=7)
    assert_almost_equal(dataseg[31, 30503], -0.79980176796607527, decimal=7)

    nw = numeigs # Number of weights, as per Fortran code
    
    # ! get the pseudoinverse of S
    # Compute the pseudo-inverse of the sphering matrix (for later use?)
    assert S.shape == (nx, nx) == (32, 32)
    # assert S[0, 0] == 0.043377346952119616
    # call DCOPY(nx*nx,S,1,Stmp2,1)
    Stmp2 = S.copy()
    Spinv = np.zeros((nx, numeigs))
    assert Spinv.shape == (nx, nw) == (32, 32)
    sUtmp = np.zeros((numeigs, numeigs))
    assert sUtmp.shape == (32, 32)
    sVtmp = np.zeros((numeigs, nx))
    assert sVtmp.shape == (32, 32)
    print(f"numeigs = {numeigs}, nw = {nw}")
    
    # call DGESVD( 'A', 'S', numeigs, nx, Stmp2, nx, eigs, sUtmp, numeigs, sVtmp, numeigs, work, lwork, info )
    sUtmp, eigs, sVtmp = np.linalg.svd(Stmp2, full_matrices=False)
    assert sUtmp.shape == (nx, nw) == (32, 32)
    assert sVtmp.shape == (nw, nx) == (32, 32)
    assert_almost_equal(abs(sUtmp[0, 0]), 0.011415317812644162)
    assert_almost_equal(abs(sUtmp[0, 1]), 0.022133957340276716)
    assert_almost_equal(abs(sVtmp[0, 0]),  0.011415317812644188)
    assert_almost_equal(abs(sVtmp[0, 1]), 0.0004865397257969421)
    assert_almost_equal(eigs[0], 0.45268334)

    sVtmp[:numeigs, :] /= eigs[:numeigs, np.newaxis]  # Normalize eigenvectors by eigenvalues
    assert_almost_equal(abs(sVtmp[0, 0]), 0.025217004224530888)
    assert_almost_equal(abs(sVtmp[31, 31]), 12.0494339875739)
    # Explicitly index to ensure the shape remains (nx, nw)
    # call DGEMM('T','T',nx,numeigs,numeigs,dble(1.0),sVtmp,numeigs,sUtmp,numeigs,dble(0.0),Spinv,nx)
    Spinv[:, :] = sVtmp.T @ sUtmp.T  # Pseudo-inverse of the sphering matrix
    assert_almost_equal(Spinv[0, 0], 33.11301219430311)

    # if (seg_rank == 0 .and. print_debug) then
    #    print *, 'S = '; call flush(6)
    #   call matout(S(1:2,1:2),2,2)
    #   print *, 'Sphered data = '; call flush(6)
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
    state, LL = _core_amica(
        X=dataseg,
        config=config,
        state=state,
        sldet=sldet,
        )
    gm = state.gm
    mu = state.mu
    rho = state.rho
    sbeta = state.sbeta
    W = state.W
    A = state.A
    c = state.c
    alpha = state.alpha
    return S, mean, gm, mu, rho, sbeta, W, A, c, alpha, LL


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

    '''n_components=n_components,
    n_models=n_models,
    n_mixtures=n_mixtures,
    max_iter=max_iter,
    pdftype=pdftype,
    do_reject=do_reject,
    tol=tol,
    lrate=lrate,
    rholrate=rholrate,
    do_newton=do_newton,
    newt_start=newt_start,
    newtrate=newtrate,
    newt_ramp=newt_ramp,'''
    assert config.n_components == 32
    assert config.n_models == 1
    assert config.n_mixtures == 3
    assert config.max_iter == 200
    assert config.pdftype == 0
    assert config.tol == 1e-7
    assert not config.do_reject
    assert config.rholrate == 0.05
    assert config.do_newton
    assert config.newt_start == 50
    assert config.newtrate == 1
    assert config.newt_ramp == 10
    assert_almost_equal(sldet, -65.935050239880198)

    # The API will use n_components but under the hood we'll match the Fortran naming
    # TODO: Maybe rename n_components to num_comps in the config dataclass?
    num_comps = config.n_components
    num_models = config.n_models
    num_mix = config.n_mixtures
    # !-------------------- ALLOCATE VARIABLES ---------------------
    print("Allocating variables ...")
    # TODO: I think this should have a num_models dimension

    assert state.A.shape == (num_comps, num_comps)
    assert_allclose(state.A, 0)

    assert state.W.shape == (num_comps, num_comps, num_models)
    assert state.W.dtype == np.float64


    # These are all passed to get_updates_and_likelihood
    
    # allocate( wr(nw),stat=ierr); call tststat(ierr); wr = dble(0.0)
    # nd = np.zeros((max(1, max_iter), num_comps), dtype=np.float64)

    assert state.gm.shape == (num_models,)

    assert state.alpha.shape == (num_comps, num_mix)

    assert state.mu.shape == (num_comps, num_mix)
    assert_allclose(state.mu, 0)

    mutmp = np.zeros((num_comps, num_mix))
    
    # if update_mu:
    
    assert state.sbeta.shape == (num_comps, num_mix)
    assert_allclose(state.sbeta, np.nan)

    # sbetatmp = np.zeros((num_comps, num_mix))  # Beta parameters
    # if update_beta:

    assert state.rho.shape == (num_comps, num_mix)
    assert_allclose(state.rho, 1.5)

    # !------------------- INITIALIZE VARIABLES ----------------------
    # print *, myrank+1, ': Initializing variables ...'; call flush(6);
    # if (seg_rank == 0) then
    print("Initializing variables ...")

    if load_gm:
        raise NotImplementedError()
    else:
        # gm[:] = int(1.0 / num_models)
        assert_allclose(state.gm.sum(), 1.0)
    if load_alpha:
        raise NotImplementedError()
    else:
        state.alpha[:, :num_mix] = 1.0 / num_mix
    if load_mu:
        raise NotImplementedError()
    else:
        values = np.arange(num_mix) - (num_mix - 1) / 2
        state.mu[:, :] = values[np.newaxis, :]
        assert state.mu.shape == (num_comps, num_mix) == (32, 3)
        np.testing.assert_allclose(state.mu[:, 0], -1.0)
        np.testing.assert_allclose(state.mu[:, 1], 0.0)
        np.testing.assert_allclose(state.mu[:, 2], 1.0)
        if not fix_init:
            mutmp = MUTMP.copy().T
            state.mu[:, :num_mix] = state.mu[:, :num_mix] + 0.05 * (1.0 - 2.0 * mutmp)
            assert_almost_equal(state.mu[0, 0], -1.0009659467356704, decimal=7)
            assert_almost_equal(state.mu[31, 2], 0.99866076686138183, decimal=7)
    if load_beta:
        raise NotImplementedError()
    else:
        if fix_init:
            raise NotImplementedError()
        else:
            state.sbeta[:, :num_mix] = 1.0 + 0.1 * (0.5 - sbetatmp)
            assert_almost_equal(state.sbeta[0, 0], 0.96533589542801645, decimal=7)
            assert_almost_equal(sbetatmp[0, 0], 0.84664104055448097)
    if load_rho:
        raise NotImplementedError()
    else:
        np.testing.assert_allclose(state.rho, 1.5)
    if load_c:
        raise NotImplementedError()
    else:
        state.c[:, :] = 0.0
        assert state.c.shape == (num_comps, num_models) == (32, 1)
    if load_A:
        raise NotImplementedError()
    else:
        comp_list, _ = get_component_indices(config.n_components, config.n_models)
        for h, _ in enumerate(range(num_models), start=1):
            h_index = h - 1
            # TODO: if A has a num_models dimension, this fancy indexing isnt needed
            # FIXME: This indexing will fail if num_models > 1
            state.A[:, (h_index)*num_comps:h*num_comps] = 0.01 * (0.5 - WTMP)
            if h == 1:
                assert_almost_equal(state.A[0, 0], 0.0041003901044031916, decimal=7)
            idx = np.arange(num_comps)
            cols = h_index * num_comps + idx
            state.A[idx, cols] = 1.0
            Anrmk = np.linalg.norm(state.A[:, cols], axis=0)
            if h == 1:
                assert_almost_equal(Anrmk[0], 1.0001205115690768)
                assert_almost_equal(Anrmk[1], 1.0001597653323635)
                assert_almost_equal(Anrmk[2], 1.0001246023020249)
                assert_almost_equal(Anrmk[3], 1.0001246214648813)
                assert_almost_equal(Anrmk[4], 1.0001391792172245)
                assert_almost_equal(Anrmk[5], 1.0001153695881879)
                assert_almost_equal(Anrmk[6], 1.0001348988545486)
                assert_almost_equal(Anrmk[31], 1.0001690977165658)
            else:
                raise ValueError("Unexpected model index")
            state.A[:, cols] /= Anrmk

            if h == 1:
                assert_almost_equal(state.A[0, 0], 0.99987950295221151)
                assert_almost_equal(state.A[0, 1], 0.0031751973942113266)
                assert_almost_equal(state.A[0, 2], 0.0032972413345084516)
                assert_almost_equal(state.A[0, 3], -0.0039658956397471655)
                assert_almost_equal(state.A[0, 4], -0.003799613000692897)
                assert_almost_equal(state.A[0, 5], 0.0028189089968969124)
                assert_almost_equal(state.A[0, 6], -0.0049667241649223011)
                assert_almost_equal(state.A[0, 7], -0.0049493288857340749)
                assert_almost_equal(state.A[0, 31], 0.0033698692262480665)

                assert_equal(comp_list[0, 0], 1)
                assert_equal(comp_list[1, 0], 2)
                assert_equal(comp_list[2, 0], 3)
                assert_equal(comp_list[3, 0], 4)
                assert_equal(comp_list[4, 0], 5)
                assert_equal(comp_list[5, 0], 6)
                assert_equal(comp_list[6, 0], 7)
                assert_equal(comp_list[31, 0], 32)
            else:
                raise ValueError("Unexpected model index")
    # end load_A
    iterating = True if "iter" in locals() else False
    W, wc = get_unmixing_matrices(
        iterating=iterating,
        c=state.c,
        A=state.A,
        comp_list=comp_list,
        W=state.W,
        num_models=num_models,
    )
    assert_almost_equal(W[0, 0, 0], 1.0000898173968631, decimal=7)
    state.W = W.copy()
    del W # safe guard against accidental use of W instead of state.W

    # load_comp_list
    # if load_comp_list:
    #    raise NotImplementedError()
    # XXX: Seems like the Fortran code duplicated this step?
    for h, _ in enumerate(range(num_models), start=1):
        h_index = h - 1
        comp_list[:, h_index] = h_index * num_comps + np.arange(1, num_comps + 1)
    if h == 1:
        assert_equal(comp_list[0, 0], 1)
        assert_equal(comp_list[1, 0], 2)
        assert_equal(comp_list[2, 0], 3)
        assert_equal(comp_list[3, 0], 4)
        assert_equal(comp_list[4, 0], 5)
        assert_equal(comp_list[5, 0], 6)
        assert_equal(comp_list[6, 0], 7)
        assert_equal(comp_list[31, 0], 32)
    else:
        raise NotImplementedError("Unexpected model index")
    
    # if (print_debug) then
    #  print *, 'data ='; call flush(6)

    # if load_rej:
    #    raise NotImplementedError()    

    # !-------------------- Determine optimal block size -------------------
    if do_opt_block:
        raise NotImplementedError()
    else:
        # call allocate_blocks
        N1 = dataseg.shape[-1] # 2 * block_size * num_thrds
        # assert N1 == 1024 # 10240
        # b = np.zeros((N1, nw, num_models))  # Allocate b
        # v = np.zeros((N1, num_models)) # posterior probability for each model
        # y = np.zeros((N1, nw, num_mix, num_models))
        # z = np.zeros((N1, nw, num_mix, num_models))  # normalized mixture responsibilities within each component
        # z0 = np.zeros((N1, num_mix))  # Allocate z0
        # z0 = np.zeros((N1, nw, num_mix)) # per-mixture evidence: mixture-weighted density for sample m, component i, mixture j
        # fp = np.zeros((N1, nw, num_mix)) # shape is (N1) in Fortran
        # ufp = np.zeros((N1, nw, num_mix)) # shape is (N1) in Fortran
        #u = np.zeros((N1, nw, num_mix)) # shape is (N1) in Fortran
        # utmp = np.zeros(N1)
        # ztmp = np.zeros((N1, nw)) # shape is (N1) in Fortran
        # vtmp = np.zeros(N1)
        # logab = np.zeros((N1, nw, num_mix)) # shape is (N1) in Fortran
        # tmpy = np.zeros((N1, nw, num_mix)) # shape is (N1) in Fortran
        # Ptmp = np.zeros((N1, num_models)) #  per-sample, per-model Log likelihood
        # git = np.zeros((N1, nw, num_models)) # Python only: per-sample, per-component, per-model LSE across mixtures.
        # P = np.zeros(N1) # Per-sample total log-likelihood across models.
        # Pmax = np.zeros(N1)
        # Pmax_br = np.zeros((N1, nw)) # Python only
        # tmpvec = np.zeros(N1)
        # tmpvec_z0 = np.zeros((N1, nw, num_mix)) # Python only
        # tmpvec_mat_dlambda = np.zeros((N1, nw, num_mix)) # Python only
        # tmpvec_fp = np.zeros((N1, nw, num_mix)) # Python only
        # tmpvec2 = np.zeros(N1)
        # tmpvec2_fp = np.zeros((N1, nw, num_mix)) # Python only
        # tmpvec2_z0 = np.zeros((N1, nw, num_mix)) # Python only
    myrank = 0
    print(f"{myrank + 1}: block size = {N1}")
    # for seg, _ in enumerate(range(numsegs), start=1):

    # v[:, :] = 1.0 / num_models

    print(f"{myrank+1} : entering the main loop ...")

    # !XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX main loop XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    state, LL = optimize(
        X=X,
        sldet=sldet,
        wc=wc,
        config=config,
        state=state,
    )
    return state, LL


@line_profiler.profile
def optimize(
        *,
        X,
        sldet,
        wc,
        config,
        state,
):
    """Main optimization loop for AMICA."""
    leave = False
    iter = 1
    numrej = 0
    N1 = X.shape[1]
    assert N1 == 30504

    n_models = config.n_models
    n_mixtures = config.n_mixtures
    num_comps = config.n_components
    pdftype = config.pdftype
    do_reject = do_reject = config.do_reject
    do_newton = config.do_newton
    newt_start = config.newt_start
    assert newt_start == 50
    num_models = n_models
    num_mix = n_mixtures
    
    # Pre-allocate all workspace buffers
    work = AmicaWorkspace()
    # Define buffer specifications for this iteration
    buffer_specs = {
        # Source and intermediate computation buffers
        "b": (N1, num_comps),                    
        "scratch_2D": (N1, num_comps), # we use this for g and a modloglik intermediate
        "y": (N1, num_comps, num_mix),           
        "z": (N1, num_comps, num_mix),
        "fp": (N1, num_comps, num_mix),
        "ufp": (N1, num_comps, num_mix),
        # Workspace arrays that need zero initialization
        "modloglik": (N1, num_models),
        "loglik": (N1,),
        "dWtmp": (num_comps, num_comps, num_models),
    }
    # Allocate most buffers as empty (faster)
    work.allocate_all(buffer_specs, init="empty")
    # Allocate buffers that need zero initialization separately
    work.allocate_buffer("modloglik", buffer_specs["modloglik"], init="zeros")
    work.allocate_buffer("dWtmp", buffer_specs["dWtmp"], init="zeros")
    # Log workspace memory usage for debugging
    total_memory_mb = work.total_memory_usage() / (1024 * 1024)
    print(f"Workspace allocated: {len(buffer_specs)} buffers, {total_memory_mb:.1f} MB total")
    
    # Initialize updates container
    updates = initialize_updates(config)
    
    # Initialize updates structure - replaces individual d*_numer/denom arrays    
    Dtemp = np.zeros(num_models, dtype=np.float64)
    # Track determinant sign per model for completeness (not used in likelihood)
    Dsign = np.zeros(num_models, dtype=np.int8)
    LL = np.zeros(max(1, config.max_iter), dtype=np.float64)  # Log likelihood


    lrate = config.lrate
    rholrate = config.rholrate
    c_start = time.time()
    c1 = time.time()
    while iter <= config.max_iter:
        # ============================== Subsection ====================================
        # === Update the unmixing matrices and compute the determinants ===
        # The Fortran code computed log|det(W)| indirectly via QR factorization
        # and summing log(abs(diag(R))). We use numpy's slogdet which is more direct.
        # Amica uses log|det(W)|, and not the sign, but we store Dsign for completeness.
        # ==============================================================================
        
        # !----- get determinants
        work.zero_all()
        metrics = IterationMetrics(
            iter=iter,
            lrate=lrate,
            rholrate=rholrate,
        )
        iter = metrics.iter


        for h, _ in enumerate(range(num_models), start=1):
            h_index = h - 1
            # Use slogdet on the original unmixing matrix to get sign and log|det|
            sign, logabsdet = compute_sign_log_determinant(
                unmixing_matrix=state.W[:, :, h_index],
                minlog=minlog,
            )
            Dtemp[h_index] = logabsdet
            Dsign[h_index] = sign        

            # Determinant computed above via slogdet (sign stored in Dsign)
            if h == 1 and iter == 1:
                assert_almost_equal(Dtemp[h - 1], 0.0044558350900245226)
            elif h == 1 and iter == 2:
                assert_almost_equal(Dtemp[h - 1], 0.0039077958090355637)
        Dsum = Dtemp.copy() # shape (num_models,)

        # ============================== Subsection ====================================
        # updates, metrics = get_updates_and_likelihood()
        
        comp_list, _ = get_component_indices(config.n_components, config.n_models)
        nw = num_comps

        W = state.W
        sbeta = state.sbeta
        rho = state.rho
        gm = state.gm
        mu = state.mu
        alpha = state.alpha

        updates.reset()

        dgm_numer = updates.dgm_numer
        dmu_numer = updates.dmu_numer
        dmu_denom = updates.dmu_denom
        dalpha_numer = updates.dalpha_numer
        dalpha_denom = updates.dalpha_denom
        dbeta_numer = updates.dbeta_numer
        dbeta_denom = updates.dbeta_denom
        drho_numer = updates.drho_numer
        drho_denom = updates.drho_denom
        dc_numer = updates.dc_numer
        dc_denom = updates.dc_denom

        if do_newton:
            dbaralpha_numer = updates.newton.dbaralpha_numer
            dbaralpha_denom = updates.newton.dbaralpha_denom
            dkappa_numer = updates.newton.dkappa_numer
            dkappa_denom = updates.newton.dkappa_denom
            dlambda_numer = updates.newton.dlambda_numer
            dlambda_denom = updates.newton.dlambda_denom
            dsigma2_numer = updates.newton.dsigma2_numer
            dsigma2_denom = updates.newton.dsigma2_denom

        # === Section: Initialize Accumulators & Buffers ===
        # Initialize arrays for likelihood computations and parameter updates
        # Set up numerator/denominator accumulators for gradient updates
        #-------------------------------------------------------------------------------
        N1 = X.shape[-1]
        assert N1 == 30504  # number of samples in data segment


        ufp = work.get_buffer("ufp")
        dWtmp = work.get_buffer("dWtmp")
        loglik = work.get_buffer("loglik")
        
        # Validate critical buffer shapes
        assert ufp.shape == (N1, num_comps, num_mix)
        assert dWtmp.shape == (num_comps, num_comps, num_models)

        # !--------- loop over the segments ----------

        if do_reject:
            raise NotImplementedError()
        else:
            pass

        # !--------- loop over the blocks ----------
        '''
        # In Fortran, the OMP parallel region would start before the lines below.
        # !$OMP PARALLEL DEFAULT(SHARED) &
        # !$OMP & PRIVATE (thrdnum,tblksize,t,h,i,j,k,xstrt,xstp,bstrt,bstp,LLinc,tmpsum,usum,vsum)
        # thrdnum = omp_get_thread_num()
        # tblksize = bsize / num_thrds
        tblksize = int(bsize / num_thrds)
        # !print *, myrank+1, thrdnum+1, ': Inside openmp code ... '; call flush(6)
        '''
        
        # === Section: E-step per Model ===
        # - Transform data via unmixing (b = W^T @ X)
        # - Evaluate source densities (per component and mixture)
        # - Aggregate mixture likelihoods with log-sum-exp
        # - Compute normalized responsibilities within each component
        #---------------------------------------------------------------------------------
        b = work.get_buffer("b")
        z = work.get_buffer("z")
        modloglik = work.get_buffer("modloglik")
        scratch = work.get_buffer("scratch_2D") # will be reused as g
        
        for h, _ in enumerate(range(num_models), start=1):
            h_index = h - 1
            comp_indices = comp_list[:, h_index] - 1
            dataseg = X
            N1, nw, num_mix = dataseg.shape[-1], config.n_components, config.n_mixtures
            fp = work.get_buffer("fp")
            y = work.get_buffer("y")

            # -- 0. Baseline terms for per-sample model log-likelihood --            
            initial = get_initial_model_log_likelihood(
                    unmixing_logdet=Dtemp[h_index],
                    whitening_logdet=sldet,
                    model_weight=gm[h_index],
            )
            # Broadcast to all samples
            modloglik[:, h_index].fill(initial)
            
            # ===========================================================================
            #                       Expectation Step (E-step)
            # ===========================================================================

            # 1. --- Compute source pre-activations
            # !--- get b
            b = compute_preactivations(
                X=dataseg,
                unmixing_matrix=state.W[:, :, h_index],
                bias=wc[:, h_index],
                do_reject=config.do_reject,
                out_activations=b,
            )

            # 2. --- Source densities, and per-sample mixture log-densities (logits) ---
            y, z = _compute_source_densities(
                pdftype=config.pdftype,
                b=b,
                sbeta=sbeta,
                mu=mu,
                alpha=alpha,
                rho=rho,
                comp_indices=comp_indices,
                out_sources=y,
                out_logits=z,
                )
            z0 = z  # log densities (alias for clarity with Fortran code)

            # 3. --- Aggregate mixture logits into per-sample model log likelihoods
            compute_per_sample_model_loglikelihood(
                log_densities=z0,
                out_modloglik=modloglik[:, h_index],
                scratch=scratch,
            )
            scratch.fill(0.0) # clear scratch for reuse
            scratch = None # guard against misuse until reassignment
            
            # 4. -- Responsibilities within each component ---
            # !--- get normalized z
            z = compute_mixture_responsibilities(log_densities=z0, inplace=True)
            z0 = None  # Let's stick with one name to avoid confusion
        # end do (h)

        # 5. --- Across-model Responsibilities and Total Log-Likelihood ---
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
        loglik = np.logaddexp.reduce(modloglik, axis=1, out=loglik) # across models
        LLinc = loglik.sum()

        for h, _ in enumerate(range(num_models), start=1):
            if do_reject:
                raise NotImplementedError()
            else:
                pass
                #--------------------------FORTRAN CODE-------------------------
                # dataseg(seg)%modloglik(h,xstrt:xstp) = Ptmp(bstrt:bstp,h)
                # dataseg(seg)%loglik(xstrt:xstp) = P(bstrt:bstp)
                #---------------------------------------------------------------
            
            #---------------Total Log-Likelihood and Model Responsibilities-----------------
            #--------------------------FORTRAN CODE-------------------------
            # v(bstrt:bstp,h) = dble(1.0) / exp(P(bstrt:bstp) - Ptmp(bstrt:bstp,h))
            #---------------------------------------------------------------
        # Model responsibilities (per sample)
        # in-place softmax over models: modloglik becomes v (responsibilities)
        # TODO: consider keeping v and modloglik separate once chunking is implemented

        # fast-path: if only one model, skip softmax and set responsibilities to 1
        if num_models == 1:
            modloglik.fill(1.0)
        else:
            modloglik = _inplace_softmax(modloglik)

        # NOTE: modloglik was mutated in-place to responsibilities (v)
        v = modloglik
        modloglik = None  # guard against use of mutated modloglik

        # if (print_debug .and. (blk == 1) .and. (thrdnum == 0)) then
        # !--- get g, u, ufp
        # !print *, myrank+1,':', thrdnum+1,': getting g ...'; call flush(6)
        g = work.get_buffer("scratch_2D")  # reuse for g workspace
        for h, _ in enumerate(range(num_models), start=1):
            h_index = h - 1
            #--------------------------FORTRAN CODE-------------------------
            # vsum = sum( v(bstrt:bstp,h) )
            # dgm_numer_tmp(h) = dgm_numer_tmp(h) + vsum 
            #---------------------------------------------------------------
            vsum = v[:, h_index].sum()
            dgm_numer[h_index] += vsum

            # if update_A:
            #--------------------------FORTRAN CODE-------------------------
            # call DSCAL(nw*tblksize,dble(0.0),g(bstrt:bstp,:),1)
            #---------------------------------------------------------------
            
            # for do (i = 1, nw)
            v_h = v[:, h_index] #  select responsibilities for this model
            if do_newton:
                #--------------------------FORTRAN CODE-------------------------
                # !print *, myrank+1,':', thrdnum+1,': getting dsigma2 ...'; call flush(6)
                # tmpsum = sum( v(bstrt:bstp,h) * b(bstrt:bstp,i,h) * b(bstrt:bstp,i,h) )
                # dsigma2_numer_tmp(i,h) = dsigma2_numer_tmp(i,h) + tmpsum
                # dsigma2_denom_tmp(i,h) = dsigma2_denom_tmp(i,h) + vsum
                #---------------------------------------------------------------
                tmpsum_A_vec = np.sum(v_h[:, None] * b ** 2, axis=0) # # shape: (nw,)
                dsigma2_numer[:, h_index] += tmpsum_A_vec
                dsigma2_denom[:, h_index] += vsum  # vsum is scalar, broadcasts to all
            elif not do_newton:
                raise NotImplementedError()
            # if update_c:
            if do_reject:
                raise NotImplementedError()
                # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,dataseg(seg)%goodinds(xstrt:xstp)) )
            else:
                #--------------------------FORTRAN CODE-------------------------
                # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,xstrt:xstp) )
                #---------------------------------------------------------------
                # # Vectorized update for dc
                assert dataseg.shape[1] == v_h.shape[0]  # should match block size
                tmpsum_c_vec = np.sum(dataseg * v_h, axis=1)
            # dc_numer_tmp(i,h) = dc_numer_tmp(i,h) + tmpsum
            # dc_denom_tmp(i,h) = dc_denom_tmp(i,h) + vsum
            dc_numer[:, h_index] += tmpsum_c_vec
            dc_denom[:, h_index] += vsum  # # vsum is scalar, broadcasts
            
            
            #--------------------------FORTRAN CODE-------------------------
            # for do (j = 1, num_mix)
            # u(bstrt:bstp) = v(bstrt:bstp,h) * z(bstrt:bstp,i,j,h)
            # usum = sum( u(bstrt:bstp) )
            #---------------------------------------------------------------
            u = v_h[:, None, None] * z  # shape: (n_samples, nw, num_mix)
            assert u.shape == (N1, nw, num_mix)
            usum = u[:, :, :].sum(axis=0)  # shape: (nw, num_mix)
            assert usum.shape == (32, 3)  # nw, num_mix
            
            assert rho.shape == (config.n_components, num_mix)
            if iter == 6 and h == 1: # and blk == 1:
                # and j == 3 and i == 1 
                assert rho[0, 2] == 1.0
            fp = compute_source_scores(
                pdftype=pdftype,
                y=y,
                rho=rho,
                comp_indices=comp_indices,
                out_scores=fp,
            )
            assert fp.shape == (N1, nw, num_mix)
            # --- Vectorized calculation of ufp and g update ---
            if iter == 1 and h == 1: # and blk == 1 
                assert g[0, 0] == 0.0
            ufp, g = accumulate_scores(
                scores=fp,
                responsibilities=u,
                scale_params=sbeta,
                comp_indices=comp_indices,
                out_ufp=ufp,
                out_g=g,
            )
            assert ufp.shape == (N1, nw, num_mix)   

            # --- Vectorized Newton-Raphson Updates ---
            if do_newton and iter >= newt_start:

                if iter == 50: # and blk == 1:
                    assert np.all(dkappa_numer == 0.0)
                    assert np.all(dkappa_denom == 0.0)

                #--------------------------FORTRAN CODE-------------------------
                # for (i = 1, nw) ... for (j = 1, num_mix)
                # tmpsum = sum( ufp(bstrt:bstp) * fp(bstrt:bstp) ) * sbeta(j,comp_list(i,h))**2
                # dkappa_numer_tmp(j,i,h) = dkappa_numer_tmp(j,i,h) + tmpsum
                # dkappa_denom_tmp(j,i,h) = dkappa_denom_tmp(j,i,h) + usum
                #---------------------------------------------------------------

                comp_indices = comp_list[:, h_index] - 1  # Shape: (nw,)
                # 1) Kappa updates (curvature terms for A)
                ufp_fp_sums = np.sum(ufp[:, :, :] * fp[:, :, :], axis=0)
                sbeta_vals = sbeta[comp_indices, :] ** 2
                tmpsum_kappa = ufp_fp_sums * sbeta_vals # Shape: (nw, num_mix)
                dkappa_numer[:, :, h_index] += tmpsum_kappa
                dkappa_denom[:, :, h_index] += usum
                
                # 2) Lambda updates
                # ---------------------------FORTRAN CODE---------------------------
                # tmpvec(bstrt:bstp) = fp(bstrt:bstp) * y(bstrt:bstp,i,j,h) - dble(1.0)
                # tmpsum = sum( u(bstrt:bstp) * tmpvec(bstrt:bstp) * tmpvec(bstrt:bstp) )
                # dlambda_numer_tmp(j,i,h) = dlambda_numer_tmp(j,i,h) + tmpsum
                # dlambda_denom_tmp(j,i,h) = dlambda_denom_tmp(j,i,h) + usum
                # ------------------------------------------------------------------
                tmpvec_mat_dlambda = (
                    fp[:, :, :] * y[:, :, :] - 1.0
                )
                tmpsum_dlambda = np.sum(
                    u[:, :, :] * np.square(tmpvec_mat_dlambda[:, :, :]), axis=0
                )  # shape: (nw, num_mix)
                dlambda_numer[:, :, h_index] += tmpsum_dlambda
                dlambda_denom[:, :, h_index] += usum
                

                # 3) (dbar)Alpha updates
                # ---------------------------FORTRAN CODE---------------------------
                # for (i = 1, nw) ... for (j = 1, num_mix)
                # dbaralpha_numer_tmp(j,i,h) = dbaralpha_numer_tmp(j,i,h) + usum
                # dbaralpha_denom_tmp(j,i,h) = dbaralpha_denom_tmp(j,i,h) + vsum
                # ------------------------------------------------------------------
                dbaralpha_numer[:, :, h_index] += usum
                dbaralpha_denom[:,:, h_index] += vsum

            # end if (do_newton and iter >= newt_start)
            elif not do_newton and iter >= newt_start:
                raise NotImplementedError()
            # end if (update_A)

            # Alpha (mixture weights)
            # if update_alpha:
            # -------------------------------FORTRAN--------------------------------
            # for (i = 1, nw) ... for (j = 1, num_mix)
            # dalpha_numer_tmp(j,comp_list(i,h)) = dalpha_numer_tmp(j,comp_list(i,h)) + usum
            # dalpha_denom_tmp(j,comp_list(i,h)) = dalpha_denom_tmp(j,comp_list(i,h)) + vsum
            # -----------------------------------------------------------------------
            # NOTE: the vectorization of this variable results in some numerical differences
            # That propogate to several other variables due to a dependency chan
            # e.g. dalpha_numer_tmp -> dalpha_numer -> alpha -> z0 etc.
            comp_indices = comp_list[:, h_index] - 1  # TODO: do this once and re-use
            dalpha_numer[comp_indices, :] += usum  # shape: (nw, num_mix)
            dalpha_denom[comp_indices, :] += vsum  # shape: (nw, num_mix)

            # Mu (location)
            # if update_mu:
            # 1. update numerator
            # -------------------------------FORTRAN--------------------------------
            # for (i = 1, nw) ... for (j = 1, num_mix)
            # tmpsum = sum( ufp(bstrt:bstp) )
            # dmu_numer_tmp(j,comp_list(i,h)) = dmu_numer_tmp(j,comp_list(i,h)) + tmpsum
            # -----------------------------------------------------------------------
            # XXX: Some error in each sum across 59 blocks
            tmpsum_mu = ufp[:, :, :].sum(axis=0)  # shape: (nw, num_mix)
            dmu_numer[comp_indices, :] += tmpsum_mu
            # 2. update denominator
            # -------------------------------FORTRAN--------------------------------
            # for (i = 1, nw) ... for (j = 1, num_mix)
            # if (rho(j,comp_list(i,h)) .le. dble(2.0)) then
            # tmpsum = sbeta(j,comp_list(i,h)) * sum( ufp(bstrt:bstp) / y(bstrt:bstp,i,j,h) )
            # dmu_denom_tmp(j,comp_list(i,h)) = dmu_denom_tmp(j,comp_list(i,h)) + tmpsum 
            # else
            # tmpsum = sbeta(j,comp_list(i,h)) * sum( ufp(bstrt:bstp) * fp(bstrt:bstp) )
            # -----------------------------------------------------------------------
            if np.all(rho[comp_indices, :] <= 2.0):
                # shape : (nw, num_mix)
                mu_denom_sum = np.sum(ufp[:, :, :] / y[:, :, :], axis=0)

                # shape (nw, num_mix)
                tmpsum_mu_denom = (sbeta[comp_indices, :] * mu_denom_sum)
                dmu_denom[comp_indices, :] += tmpsum_mu_denom  # XXX: Errors accumulate across 59 additions
            else:
                raise NotImplementedError()
                # Let's tackle this when we actually hit this with some data
                # So that we can compare the result against the Fortran output.
            # end if (update mu)

            # Beta (scale/precision)
            # if update_beta:
            # 1. update numerator
            # -------------------------------FORTRAN--------------------------------
            # dbeta_numer_tmp(j,comp_list(i,h)) = dbeta_numer_tmp(j,comp_list(i,h)) + usum
            # dbeta_numer_tmp[j - 1, comp_list[i - 1, h - 1] - 1] += usum
            # ----------------------------------------------------------------------
            dbeta_numer[comp_indices, :] += usum  # shape: (nw, num_mix)
            # 2. update denominator
            # -------------------------------FORTRAN--------------------------------
            # if (rho(j,comp_list(i,h)) .le. dble(2.0)) then
            # tmpsum = sum( ufp(bstrt:bstp) * y(bstrt:bstp,i,j,h) )
            # dbeta_denom_tmp(j,comp_list(i,h)) =  dbeta_denom_tmp(j,comp_list(i,h)) + tmpsum
            # ----------------------------------------------------------------------
            if np.all(rho[comp_indices, :] <= 2.0):
                tmpsum_dbeta_denom = np.sum(
                    ufp[:, :, :] * y[:, :, :], axis=0
                )
                dbeta_denom[comp_indices, :] += tmpsum_dbeta_denom  # shape: (nw, num_mix)
            else:
                raise NotImplementedError()
            # end if (update beta)
            
            # Rho (shape parameter of generalized Gaussian)
            if dorho:
                # -------------------------------FORTRAN--------------------------------
                # for (i = 1, nw) ... for (j = 1, num_mix)
                # tmpy(bstrt:bstp) = abs(y(bstrt:bstp,i,j,h))
                # logab(bstrt:bstp) = log(tmpy(bstrt:bstp))
                # tmpy(bstrt:bstp) = exp(rho(j,comp_list(i,h))*logab(bstrt:bstp))
                # logab(bstrt:bstp) = log(tmpy(bstrt:bstp))
                # ----------------------------------------------------------------------
                # 1. log of absolute value
                tmpy = np.empty((N1, nw, num_mix))
                logab = np.empty((N1, nw, num_mix)) # shape is (N1) in Fortran
                np.abs(y[:, :, :], out=tmpy)
                np.log(tmpy[:, :, :], out=logab)
                # 2. Exponentiation with rho
                rho_vals = rho[comp_indices, :]  # shape: (nw, num_mix)
                rho_vals_br = rho_vals[np.newaxis, :, :]  # shape: (1, nw, num_mix)
                # shape: (max_block_size, nw, num_mix)
                np.multiply(rho_vals_br, logab, out=logab)
                np.exp(logab, out=tmpy)  # now tmpy is exp(rho * log|y|)
                # shape: (max_block_size, nw, num_mix)
                np.log(tmpy, out=logab)
                
                # -------------------------------FORTRAN--------------------------------
                # where (tmpy(bstrt:bstp) < epsdble)
                        #    logab(bstrt:bstp) = dble(0.0)
                        # end where
                # ----------------------------------------------------------------------
                # Set values below epsdble to 0.0
                logab[tmpy < epsdble] = 0.0
                # -------------------------------FORTRAN--------------------------------
                # logab[bstrt-1:bstp][tmpy[bstrt-1:bstp] < epsdble] = 0.0
                # tmpsum = sum( u(bstrt:bstp) * tmpy(bstrt:bstp) * logab(bstrt:bstp) )
                # drho_numer_tmp(j,comp_list(i,h)) =  drho_numer_tmp(j,comp_list(i,h)) + tmpsum
                # drho_denom_tmp(j,comp_list(i,h)) =  drho_denom_tmp(j,comp_list(i,h)) + usum
                # ----------------------------------------------------------------------
                #tmpsum = np.sum(u[bstrt-1:bstp] * tmpy[bstrt-1:bstp] * logab[bstrt-1:bstp])
                tmpsum_prod = np.sum(
                    u[:, :, :]
                    * tmpy[:, :, :]
                    * logab[:, :, :]
                , axis=0
                )
                drho_numer[comp_indices, :] += tmpsum_prod
                drho_denom[comp_indices, :] += usum
                
                if np.any(rho[comp_indices, :] > 2.0):
                    raise NotImplementedError()
            elif not dorho:
                raise NotImplementedError()

            # if (print_debug .and. (blk == 1) .and. (thrdnum == 0)) then
            # if update_A:
            #--------------------------FORTRAN CODE-------------------------
            # call DSCAL(nw*nw,dble(0.0),Wtmp2(:,:,thrdnum+1),1)   
            # call DGEMM('T','N',nw,nw,tblksize,dble(1.0),g(bstrt:bstp,:),tblksize,b(bstrt:bstp,:,h),tblksize, &
            #            dble(1.0),Wtmp2(:,:,thrdnum+1),nw)
            # call DAXPY(nw*nw,dble(1.0),Wtmp2(:,:,thrdnum+1),1,dWtmp(:,:,h),1)
            #---------------------------------------------------------------
            # Wtmp2 has a 3rd dimension for threads in Fortran
            Wtmp2 = np.dot(g.T, b[:, :]) #  # shape (num_comps, num_comps)
            dWtmp[:, :, h - 1] += Wtmp2
        # end do (h)
        # end do (blk)'
        if iter == 1: # and blk == 59:
            # j is 3 and i is 32 by this point
            # assert j == 3
            # assert i == 32
            assert h == 1
            assert_allclose(rho, 1.5)
            assert_almost_equal(g[-808, 0], 0.19658642673900478)
            assert_almost_equal(g[-1, 31], -0.22482758905985217)
            assert dgm_numer[0] == 30504
            # XXX: this gets explicitly tested against tmpsum_prod in the dorho block.
            # assert_almost_equal(tmpsum, -52.929467835976844)
            assert dsigma2_denom[31, 0] == 30504
            assert_almost_equal(dsigma2_numer[31, 0], 30521.3202213734, decimal=6) # XXX: watch this
            assert_almost_equal(dsigma2_numer[0, 0], 30517.927488143538, decimal=6)
            assert_almost_equal(dc_numer[31, 0], 0)
            assert dc_denom[31, 0] == 30504
            assert_allclose(v, 1)
            assert_almost_equal(z[-808, 31, 2], 0.72907838295502048)
            assert_almost_equal(z[-1, 31, 2], 0.057629436774909774)
            # assert_almost_equal(z0[-808, 31, 2], -1.7145368856186436)
            assert_almost_equal(u[-808, 31, 2], 0.72907838295502048)
            assert_almost_equal(u[-1, 31, 2], 0.057629436774909774)
            # assert_almost_equal(tmpvec_fp[-808, 31, 2], -2.1657430925146017)
            # assert_almost_equal(tmpvec2_fp[-1, 31, 2], 1.3553626849082627)
            assert_almost_equal(fp[-808, 31, 2], 0.50793264023957352)
            assert_almost_equal(ufp[-808, 31, 2], 0.37032270799594241)
            assert_almost_equal(dalpha_numer[31, 2], 9499.991274464508, decimal=5)
            assert dalpha_denom[31, 2] == 30504
            assert_almost_equal(dmu_numer[31, 2], -3302.4441649143237, decimal=5) # XXX: test another indice since this is numerically unstable
            assert_almost_equal(dmu_numer[0, 0], 6907.8603204569654, decimal=5)
            assert_almost_equal(sbeta[31, 2], 1.0138304802882583)
            assert_almost_equal(dmu_denom[31, 2], 28929.343372016403, decimal=2) # XXX: watch this for numerical stability
            assert_almost_equal(dmu_denom[0, 0], 22471.172722479747, decimal=3)
            assert_almost_equal(dbeta_numer[31, 2], 9499.991274464508, decimal=5)
            assert_almost_equal(dbeta_denom[31, 2], 8739.8711658999582, decimal=6)
            assert_almost_equal(dbeta_numer[0, 0], 8967.4993064961727, decimal=5)
            assert_almost_equal(dbeta_denom[0, 0], 10124.98913119294, decimal=5)
            assert_almost_equal(y[-1, 31, 2], -1.8370080076417346)
            assert_almost_equal(logab[-808, 31,2], -3.2486146387719028)
            assert_almost_equal(tmpy[-808, 31, 2], 0.038827961341319203)
            assert_almost_equal(drho_numer[31, 2], 469.83886293477855, decimal=5)
            assert_almost_equal(drho_denom[31, 2], 9499.991274464508, decimal=5)
            assert_almost_equal(drho_numer[0, 0], 2014.2985887030379, decimal=5)
            assert_almost_equal(drho_denom[0, 0], 8967.4993064961727, decimal=5)
            assert_almost_equal(dWtmp[31, 0, 0], 143.79140032913983, decimal=6)
            assert_almost_equal(loglik[-1], -111.60532918598989)
            assert_almost_equal(LLinc, -3429802.6457936931, decimal=5) # XXX: check this value after some iterations
            # assert_almost_equal(LLinc, -89737.92559533281, decimal=6)
        elif iter == 2:
            assert_almost_equal(dc_numer[31, 0], 0)
            assert_almost_equal(dc_numer[0, 0],  0)
            assert dc_denom[31, 0] == 30504
            assert dc_denom[0, 0] == 30504
            assert_almost_equal(dalpha_numer[0, 0], 7861.9637766408878, decimal=5)
            assert_almost_equal(dalpha_numer[31, 2], 9221.7061911138153, decimal=4)
            assert dalpha_denom[31, 2] == 30504
            assert dalpha_denom[0, 0] == 30504
            assert_almost_equal(dmu_numer[0, 0], 3302.9474389348984, decimal=4)
            assert_almost_equal(dmu_denom[0, 0], 25142.015091515364, decimal=1)
            assert_almost_equal(dbeta_numer[0, 0], 7861.9637766408878, decimal=5)
            assert_almost_equal(dbeta_denom[0, 0], 6061.5281979061665, decimal=5)
            assert_almost_equal(drho_numer[0, 0], 23.719323447428629, decimal=5)
            assert_almost_equal(drho_denom[0, 0], 7861.9637766408878, decimal=5)  
            assert_almost_equal(g[-808, 0], 0.92578280732700213)
            assert_almost_equal(g[-1, 31], -0.57496468258661515)
            assert_almost_equal(u[-808, 31, 2], 0.71373487258192514)
            assert_allclose(v, 1)
            assert_almost_equal(y[-1, 31, 2], -1.8067399375706592)
            assert_almost_equal(z[-808, 31, 2], 0.71373487258192514)
            assert_almost_equal(tmpy[-808, 31, 2], 0.12551286724858962)
            assert_almost_equal(logab[-808, 31, 2], -2.075346997788714)
            #assert_almost_equal(tmpvec_fp[-808,31,2], -1.3567967124454048)
            # assert_almost_equal(tmpvec2_fp[-1,31,2], 1.3678868714057633)
            assert_almost_equal(loglik[-1], -109.77900836816768, decimal=6)
            assert_almost_equal(dWtmp[31, 0, 0], 264.40460848250513, decimal=5)
            assert_almost_equal(LLinc, -3385986.7900999608, decimal=3) # XXX: check this value after some iterations

        # In Fortran, the OMP parallel region is closed here
        # !$OMP END PARALLEL
        # !print *, myrank+1,': finished segment ', seg; call flush(6)
        
        # XXX: Later we'll figure out which variables we actually need to return
        # For now literally return any variable that has been assigned or updated
        # In alphabetical order
        likelihood, ndtmpsum, dAK, no_newt = accum_updates_and_likelihood(
            config=config,
            updates=updates,
            state=state,
            dWtmp=dWtmp,
            LLtmp=LLinc,
            iter=iter
        )
        metrics.loglik = likelihood
        metrics.ndtmpsum = ndtmpsum
        metrics.no_newt = no_newt
        updates.dAK = dAK
        # return updates, metrics

        # ==============================================================================
        ndtmpsum = metrics.ndtmpsum
        LL[iter - 1] = metrics.loglik
        # init
        startover = False
        numincs = 0
        numdecs = 0
        # XXX: checking get_updates_and_likelihood set things globally
        # This should also give an idea of the vars that are assigned within that function.
        # Iteration 1 checks that are values were set globally and are correct form baseline
        if iter == 1:
            assert_allclose(state.rho, 1.5)
            
            assert_almost_equal(state.sbeta[31, 2], 1.0138304802882583)
            # assert_almost_equal(Wtmp2[31,31, 0], 260.86288741506081, decimal=6)
            # assert_almost_equal(LLinc, -89737.92559533281, decimal=6)
            
            # These shouldnt get updated until the start of newton_optimization

            # These should also not change until the start of newton_optimization

            # accum_updates_and_likelihood checks..
            # This should also give an idea of the vars that are assigned within that function.
            assert metrics.no_newt is False
            assert_almost_equal(ndtmpsum, 0.057812635452922263)
            assert_almost_equal(LL[0], -3.5136812444614773)
        # Iteration 2 checks that our values were set globablly and updated based on the first iteration
        elif iter == 2:
            assert_almost_equal(state.rho[0, 0], 1.4573165687688203)
            assert_almost_equal(state.sbeta[31, 2], 1.0736514759262248)
            # assert_almost_equal(Wtmp2[31,31, 0], 401.76754944355537, decimal=5)
            # assert P[808] == 0.0

            # accum_updates_and_likelihood checks..
            assert metrics.no_newt is False
            assert_almost_equal(ndtmpsum, 0.02543823967703519)
            # assert_almost_equal(Wtmp[0, 0], 0.32349815400356108)
            assert_almost_equal(LL[1], -3.4687938365664754)
        # Iteration 6 was the first iteration with a non-zero value of `fp` bc rho[idx, idx] == 1.0 instead of 1.5
        elif iter == 6:
            assert_almost_equal(state.rho[0, 0], 1.6596808063060098, decimal=6)
            assert_almost_equal(LL[5], -3.4553318810532221, decimal=6)
        # iteration 13 was the first iteration with rho[idx, idx] == 2.0 instead of 1.5 or 1.0
        elif iter == 13:
            assert state.rho[0, 0] == 2
            assert_almost_equal(LL[12], -3.4479767404904833, decimal=6)
        # Iteration 49 is the last iteration before Newton optimization starts
        elif iter == 49:
            assert_almost_equal(LL[48], -3.4413377213179359, decimal=5)
        elif iter == 50:
            # This is the first iteration with newton optimization.
            assert_almost_equal(LL[49], -3.441215133563345, decimal=5)

            # This is the first iteration with newton optimization.
            #assert_almost_equal(dkappa_denom[2,31,0], 8873.0781815692208, decimal=0)
        elif iter == 51:
            # accum_updates_and_likelihood checks..
            assert_almost_equal(LL[50], -3.4410166239008801, decimal=5) # At least this is close to the Fortran output!
             
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
        if (iter <= restartiter and np.isnan(LL[iter - 1])):
            if numrestarts > maxrestarts:
                leave = True
                raise RuntimeError()
            else:
                raise NotImplementedError()
        # end if
        if iter == 2:
            assert not np.isnan(LL[iter - 1])
            assert not (LL[iter - 1] < LL[iter - 2])
        if iter > 1:
            if np.isnan(LL[iter - 1]) and iter > restartiter:
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
                    assert 1 == 0
                    if numincs > maxincs:
                        leave = True
                        print(
                            f"Exiting because likelihood increasing by less than {min_dll} "
                            f"for more than {maxincs} iterations ..."
                            )
                        assert 1 == 0
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
            assert 1 == 0  # Stop to check that exit condition is correct
            exit()
        if startover:
            raise NotImplementedError()
        else:
            # !----- do updates: gm, alpha, mu, sbeta, rho, W
            # the updated lrate & rholrate for the next iteration
            lrate, rholrate, state, wc = update_params(
                config=config,
                state=state,
                updates=updates,
                metrics=metrics,
                wc=wc,
            )
            if iter == 1:
                # XXX: making sure all variables were globally set.
                # assert_almost_equal(Anrmk[-1], 0.98448954017506363)
                assert state.gm[0] == 1
                assert_almost_equal(state.alpha[0, 0], 0.29397781623708935, decimal=5)
                assert_almost_equal(state.c[0, 0], 0.0)
                assert posdef is True
                assert_almost_equal(config.lrate, 0.05)
                assert_almost_equal(lrate, 0.05)
                assert_almost_equal(config.rholrate, 0.05)
                assert_almost_equal(rholrate, 0.05)
                assert_almost_equal(sbetatmp[0, 0], 0.90848309104731939)
                assert maxrho == 2
                assert minrho == 1
                assert_almost_equal(state.rho[0, 0], 1.4573165687688203)
                assert not state.rho[state.rho == minrho].any()
                assert_almost_equal(state.A[31, 31], 0.99984153789378194)
                assert_almost_equal(state.sbeta[31, 0], 0.97674982753812623)
                assert_almost_equal(state.mu[31, 0], -0.8568024781696123)
                assert_almost_equal(state.W[0, 0, 0], 1.0000820892004447)
                assert_almost_equal(wc[0, 0], 0)
            elif iter == 2:
                # assert_almost_equal(Anrmk[-1], 0.99554375802233519)
                assert state.gm[0] == 1
                assert_almost_equal(state.alpha[0, 0], 0.25773550277474716)
                assert_almost_equal(state.c[0, 0], 0.0)
                assert posdef is True
                assert_almost_equal(config.lrate, 0.05)
                assert_almost_equal(lrate, 0.05)
                assert_almost_equal(config.rholrate, 0.05)
                assert_almost_equal(rholrate, 0.05)
                assert_almost_equal(sbetatmp[0, 0], 1.0583363176203351)
                assert maxrho == 2
                assert minrho == 1
                assert_almost_equal(state.rho[0, 0], 1.5062036957555023)
                assert_almost_equal(state.A[31, 31], 0.99985752877785194)
                assert_almost_equal(state.sbeta[0, 0], 1.07570700640128)
                assert_almost_equal(state.mu[0, 0], -0.53783126597732789)
                assert_almost_equal(state.W[0, 0, 0], 1.0002289118030874)
                assert_almost_equal(wc[0, 0], 0)

            # if ((writestep .ge. 0) .and. mod(iter,writestep) == 0) then

            # !----- write history if it's a specified step
            # if (do_history .and. mod(iter,histstep) == 0) then

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
        comp_list,
        W,
        num_models,
        ):
    """Get unmixing matrices for AMICA."""
    if not iterating:
        # ugly hack to check if we are in the first iteration
        # All these shoudl be true first time get_unmixing_matrices is called which is before the iteration starts
        assert_almost_equal(A[0, comp_list[0, 0] - 1], 0.99987950295221151)
        assert_almost_equal(A[2, comp_list[2, 0] - 1], 0.99987541322177442)

    wc = np.zeros_like(c)
    for h, _ in enumerate(range(num_models), start=1):

        #--------------------------FORTRAN CODE-------------------------
        # call DCOPY(nw*nw,A(:,comp_list(:,h)),1,W(:,:,h),1)
        #---------------------------------------------------------------
        # TODO: assign on the fly without passing in W
        W[:, :, h - 1] = A[:, comp_list[:, h - 1] - 1].copy()
        if not iterating:
            assert_almost_equal(W[0, 0, h - 1], 0.99987950295221151)
            assert_almost_equal(W[2, 2, h - 1], 0.99987541322177442)
            assert_almost_equal(W[31, 31, h - 1], 0.99983093087263752)

        #--------------------------FORTRAN CODE-------------------------
        # call DGETRF(nw,nw,W(:,:,h),nw,ipivnw,info)
        # call DGETRI(nw,W(:,:,h),nw,ipivnw,work,lwork,info)
        #---------------------------------------------------------------
        try:
            W[:, :, h - 1] = linalg.inv(W[:, :, h - 1])
        except linalg.LinAlgError as e:
            # This issue would originate with matrix A
            # we should review the code and provide a more user friendly error message
            # if A is singular. e.g. the "weights matrix is singular or something"
            print(f"Matrix W[:,:,{h-1}] is singular!")
            raise e

        if not iterating:
            assert_almost_equal(W[0, 0, 0], 1.0000898173968631)
            assert_almost_equal(W[5, 10, 0], 0.00024088142376377521)
            assert_almost_equal(W[5, 30, 0], 0.00045275279060370794)
            assert_almost_equal(W[15, 29, 0], -0.0012173272878070241)
            assert_almost_equal(W[25, 5, 0], 0.0022362764540665224)
            assert_almost_equal(W[25, 15, 0], 0.0048541279923070843, decimal=7)
            assert_almost_equal(W[31, 31, 0], 1.0001435790123032, decimal=7)

        #--------------------------FORTRAN CODE-------------------------
        # call DGEMV('N',nw,nw,dble(1.0),W(:,:,h),nw,c(:,h),1,dble(0.0),wc(:,h),1)
        #---------------------------------------------------------------
        wc[:, h - 1] = W[:, :, h - 1] @ c[:, h - 1]
        if not iterating:
            assert_allclose(wc[:, 0], 0)
    return W, wc

def get_seg_list(raw):
    """This is a temporary function that somehwat mirrors the Fortran get_seg_list"""
    blocks_in_sample = raw.n_times  # field_dim
    num_samples = 1  # num_files
    all_blks = blocks_in_sample * num_samples
    # We'll stop here for now. and port more of the Fortran function as we need it.
    return blocks_in_sample, num_samples, all_blks


def get_updates_and_likelihood(
    X,
    *,
    config,
    state,
    updates,
    metrics,
    work,
    sldet,
    Dsum,
    wc,
):
    """Get updates and likelihood for AMICA.
    
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
        comp_indices,
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
    comp_indices : np.ndarray
        Array of shape (n_features,) containing component indices for the current model.
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
        unmixing_matrix: Weights2D,
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
    if model_weight <= 0:
        raise ValueError(f"model_weight must be > 0, got {model_weight}")  # pragma no cover noqa: E501
    #--------------------------FORTRAN CODE-------------------------
    # Ptmp(bstrt:bstp,h) = Dsum(h) + log(gm(h)) + sldet
    #---------------------------------------------------------------
    initial_modloglik = unmixing_logdet + np.log(model_weight) + whitening_logdet
    return initial_modloglik


def compute_preactivations(
        *,
        X: Data2D,  # (n_features, n_samples)
        unmixing_matrix: Weights2D,  # (n_components, n_features)
        bias: Bias1D,  # (n_components,)
        do_reject=False, 
        out_activations: Optional[Sources2D] = None,  # (n_samples, n_components)
) -> Sources2D:
    """Compute source pre-activations b[t, :] = X_t^T @ W^T - wc for model h.
    
    Parameters
    ----------
    X : array, shape (n_features, n_samples)
        Data matrix
    unmixing_matrix : array, shape (n_components, n_features)
        Unmixing matrix weights (W) for a single model h, that maps data to sources.
    bias : array, shape (n_components,)
        Weight correction (wc) vector for model h e.g. wc[:, h_index]
    do_reject : bool
        Whether to perform rejection. Currently not implemented.
    out_activations : array, shape (T, N)
        Output buffer for pre-activations (b). If None, a new array is allocated.
        This array is modified in-place.
    
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
    if out_activations is None:
        n_components = W.shape[0]
        n_samples = X.shape[1]
        out_activations = np.empty((n_samples, n_components), dtype=np.float64)
    b = out_activations
    assert wc.ndim == 1, f"wc must be 1D, got {wc.ndim}D"
    assert W.ndim == 2, f"W must be 2D, got {W.ndim}D"
    assert dataseg.ndim == 2, f"dataseg must be 2D, got {dataseg.ndim}D"
    assert b.ndim == 2, f"b must be 2D, got {b.ndim}D"

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
        # np.einsum("ft,cf->tc", dataseg, W)
        np.matmul(dataseg.T, W.T, out=b)
    # end else
    # Subtract the weight correction factor
    # TODO: when chunking, we may need to init bias first.
    b -= wc
    return b


def _compute_source_densities(
        *,
        pdftype: int,
        b: Sources2D,
        sbeta: Params2D,
        mu: Params2D,
        alpha: Params2D,
        rho: Params2D,
        comp_indices: Index1D, # TODO: pass in the pre-indexed arrays instead
        out_sources: Optional[Sources3D] = None,
        out_logits: Optional[Sources3D] = None,
        ) -> Tuple[Sources3D, Sources3D]:
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
    comp_indices : np.ndarray
        Array containing component indices for the current model.
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
    nw = len(comp_indices)
    num_mix = alpha.shape[1]
    
    # Shape assertions for new dimension standard
    # These parameters are the full arrays, not indexed yet
    assert alpha.shape[0] >= nw, f"alpha.shape[0]={alpha.shape[0]} must be >= nw={nw}"
    assert alpha.shape[1] == num_mix, f"alpha.shape[1]={alpha.shape[1]} != num_mix={num_mix}"
    assert sbeta.shape == alpha.shape, f"sbeta shape {sbeta.shape} != alpha shape {alpha.shape}"
    assert mu.shape == alpha.shape, f"mu shape {mu.shape} != alpha shape {alpha.shape}"
    assert rho.shape == alpha.shape, f"rho shape {rho.shape} != alpha shape {alpha.shape}"
    assert b.shape == (N1, nw), f"b shape {b.shape} != (N1={N1}, nw={nw})"

    if out_sources is None:
        out_sources = np.empty((N1, nw, num_mix))
    if out_logits is None:
        out_logits = np.empty((N1, nw, num_mix))
    
    # We have 3 possible log-probability functions
    def generalized_gaussian_logprob(sources, log_alpha, log_sbeta, rho, out_logits):
        """log p(y) = log(alpha) + log(sbeta) - |y|^rho - log( Gamma(1+1/rho) ) + log(2)"""
        # log(|y|)
        np.abs(sources, out=out_logits)
        np.log(out_logits, out=out_logits)
        # |y|^rho
        np.exp(rho * out_logits, out=out_logits)
        # log(alpha) + log(sbeta) - |y|^rho
        np.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        # gammaln(1 + 1/rho)
        gamma_log = gammaln(1.0 + 1.0 / rho)
        # penalty term: log(alpha) + log(sbeta) - |y|^rho - gammaln(1 + 1/rho) + log(2)
        np.subtract(out_logits, gamma_log + LOG_2, out=out_logits)
        return out_logits

    def laplacian_logprob(sources, log_alpha, log_sbeta, out_logits):
        """log p(y) = log(alpha) + log(sbeta) - |y| - log(2)"""
        np.abs(sources, out=out_logits)
        np.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        np.subtract(out_logits, LOG_2, out=out_logits)
        return out_logits

    def gaussian_logprob(sources, log_alpha, log_sbeta, out_logits):
        """log p(y) = log(alpha) + log(sbeta) - y^2 - log(sqrt(pi))"""
        np.square(sources, out=out_logits)
        np.subtract(log_alpha + log_sbeta, out_logits, out=out_logits)
        np.subtract(out_logits, LOG_SQRT_PI, out=out_logits)
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
        alpha_h = alpha[comp_indices, :]
        rho_h = rho[comp_indices, :] # All mixtures, components for this model
        sbeta_h = sbeta[comp_indices, :]      # Shape: (nw, num_mix)
        mu_h = mu[comp_indices, :]            # Shape: (nw, num_mix)
        
        # 1. Center and scale the source estimates (In-place)
        np.subtract(b[:, :, None], mu_h[None, :, :], out=out_sources)
        np.multiply(sbeta_h, out_sources, out=out_sources)
        
        #------------------Mixture Log-Likelihood for each component----------------

        #--------------------------FORTRAN CODE-------------------------
        # if (rho(j,comp_list(i,h)) == dble(1.0)) then
        # else if (rho(j,comp_list(i,h)) == dble(2.0)) then
        # z0(bstrt:bstp,j) = log(alpha(j,comp_list(i,h))) + ...
        #---------------------------------------------------------------
        # Precompute logs (reused in all 3 logprob functions)
        log_mixture_weights = np.log(alpha_h)  # shape: (nw, num_mix)
        log_scales = np.log(sbeta_h)           # shape: (nw, num_mix)

        # Masks: Laplacian (rho==1), Gaussian (rho==2); generalized Gaussian otherwise
        lap_mask = (np.isclose(rho_h, 1.0, atol=1e-12))
        gau_mask = (np.isclose(rho_h, 2.0, atol=1e-12))

        # Default: generalized Gaussian log-prob + log mixture weight 
        # This is all or the vast majority of values, so just compute it over all
        # and then overwrite where needed
        out_logits = generalized_gaussian_logprob(
            sources=out_sources,
            log_alpha=log_mixture_weights,
            log_sbeta=log_scales,
            rho=rho_h,
            out_logits=out_logits,
        )
        assert out_logits.shape == (N1, nw, num_mix)
    
        # Overwrite with Laplacian/Gaussian log-prob + log mixture weight where needed
        if lap_mask.any():
            # In-place overwrite
            laplacian_logprob(
                sources=out_sources[:, lap_mask],
                log_alpha=log_mixture_weights[lap_mask],
                log_sbeta=log_scales[lap_mask],
                out_logits=out_logits[:, lap_mask]
            )
        if gau_mask.any():
            # In-place overwrite
            gaussian_logprob(
                sources=out_sources[:, gau_mask],
                log_alpha=log_mixture_weights[gau_mask],
                log_sbeta=log_scales[gau_mask],
                out_logits=out_logits[:, gau_mask]
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


def compute_per_sample_model_loglikelihood(
        *,
        log_densities: Sources3D,
        out_modloglik: Optional[Samples1D] = None,
        scratch: Optional[Sources2D] = None,
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
    scratch : array, shape (N1, nw)
        Scratch array for intermediate computations. If None, a new array is
        allocated. This array is mutated in-place and then discarded.
    """
    assert log_densities.ndim == 3, f"log_densities must be 3D, got {log_densities.ndim}D"
    N1 = log_densities.shape[0]
    nw = log_densities.shape[1]
    if out_modloglik is None:
        out_modloglik = np.zeros((N1,), dtype=np.float64)
    if scratch is None:
        scratch = np.zeros((N1, nw), dtype=np.float64)
    assert out_modloglik.shape == (N1,)
    assert scratch.shape == (N1, nw)
    # Alias for clarity with Fortran code
    z0 = log_densities

    #--------------------------FORTRAN CODE-------------------------
    # Pmax(bstrt:bstp) = maxval(z0(bstrt:bstp,:),2)
    # z(bstrt:bstp,i,j,h) = dble(1.0) / exp(tmpvec(bstrt:bstp) - z0(bstrt:bstp,j))
    #---------------------------------------------------------------
    # NOTE that the scratch array that was passe in will also be used for the g update.
    # TODO: consider keeping g and z separate once chunking is implemented
    component_loglik = np.logaddexp.reduce(z0, axis=-1, out=scratch) # across mixtures
    out_modloglik += component_loglik.sum(axis=-1) # across components
    return out_modloglik

def _inplace_softmax(arr):
    # Stable in-place softmax along last axis.
    assert arr.ndim >= 1
    assert arr.shape[-1] >= 1
    arr -= arr.max(axis=-1, keepdims=True) 
    np.exp(arr, out=arr)
    arr /= arr.sum(axis=-1, keepdims=True)
    return arr  

def generalized_gaussian_logprob(
    *,
    sources: Sources3D,
    rho: Params2D,
    sbeta: Params2D,
    alpha: Params2D,
    out_logits: Optional[Sources3D] = None,
):
    """
    Compute generalized Gaussian log-probability.

    log(|y|) = log(alpha) + log(sbeta) - |y|^rho - log( Gamma(1+1/rho) ) + log(2)
    
    """
    if out_logits is None:
        out_logits = np.empty_like(sources)

    # 1. Prepare all parameters for broadcasting to shape
    log_mix_weight = np.log(alpha[None, :, :])  # log(alpha) shape: (1, nw, num_mix)
    log_scale = np.log(sbeta[None, :, :])       # log(sbeta) shape: (1, nw, num_mix)
    rho_br = rho[None, :, :]                    # rho shape: (1, nw, num_mix)

    # 2. Compute in steps to avoid creating large temporary arrays
    # log(|y|)
    np.abs(sources, out=out_logits)
    np.log(out_logits, out=out_logits)
    # |y|^rho
    np.exp(rho_br * out_logits, out=out_logits)
    # log(alpha) + log(sbeta) - |y|^rho
    np.subtract(log_mix_weight + log_scale, out_logits, out=out_logits)
    
    # gammaln(1 + 1/rho)
    gamma_log = gammaln(1.0 + 1.0 / rho_br)
    # penalty term: log(alpha) + log(sbeta) - |y|^rho - gammaln(1 + 1/rho) + log(2)
    np.subtract(out_logits, gamma_log + LOG_2, out=out_logits)
    return out_logits


def compute_mixture_responsibilities(
        *,
        log_densities: Sources3D,
        inplace: Literal[True, False] = True
        ) -> Sources3D:
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
        Each slice [:, i, :] sums to 1 over axis=-1 (mixtures).
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
        z.fill(1.0)
    else:
        z = _inplace_softmax(z)
    return z


def compute_source_scores(
        *,
        pdftype: int,
        y: Sources3D,
        rho: Params2D,
        comp_indices: Index1D,
        out_scores: Optional[Sources3D] = None,
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
    comp_indices : np.ndarray
        Array containing component indices for the current model.
    out_scores : Sources3D, optional
        Buffer to write the computed score function into. Shape (n_samples, n_components,
        n_mixtures). If None, a new array is allocated. This array is modified in-place.
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
    assert len(comp_indices) == nw, f"len(comp_indices)={len(comp_indices)} != nw={nw}"
    
    if out_scores is None:
        out_scores = np.empty((N1, nw, num_mix))
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
        rho_h = rho[comp_indices, :]

        # Masks: Laplacian (rho==1), Gaussian (rho==2); generalized Gaussian otherwise
        lap_mask = (np.isclose(rho_h, 1.0, atol=1e-12))
        gau_mask = (np.isclose(rho_h, 2.0, atol=1e-12))

        # Default: generalized Gaussian score function        
        # Step 1. Compute |y|^(rho_h - 1) in-place
        np.abs(y, out=out_scores)                  # out_scores = |y|
        np.log(out_scores, out=out_scores)         # log(|y|)
        np.multiply(rho_h - 1.0, out_scores, out=out_scores)
        np.exp(out_scores, out=out_scores)         # |y|^(rho_h - 1)

        # Step 2. Multiply by rho_h and sign(y) without np.sign allocation
        out_scores *= rho_h * np.where(y >= 0, 1.0, -1.0)
        
        # Overwrite with Laplacian/Gaussian score function where needed
        if lap_mask.any():
            # In-place overwrite
            np.sign(y[:, lap_mask], out=out_scores[:, lap_mask])
        if gau_mask.any():
            # In-place overwrite
            np.multiply(y[:, gau_mask], 2.0, out=out_scores[:, gau_mask])
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
        comp_indices,
        out_ufp,
        out_g,
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
    out_ufp : np.ndarray
        The output updates (ufp) of shape (n_components, n_features).
        Modified in place.
    out_g : np.ndarray
        The output gradients (g) of shape (n_components, n_features).
        Modified in place.
    
    Returns
    -------
    out_ufp : np.ndarray
        elementwise product responsibilities * scores, shape (n_samples, n_components, n_mixtures), modified in place.
    out_g : np.ndarray
        accumulated per-sample per-component score weighted by sbeta, shape (n_samples, n_components), modified in place.
    """
    # Shape assertions for new dimension standard
    N1, nw, num_mix = scores.shape
    assert responsibilities.shape == scores.shape, f"responsibilities shape {responsibilities.shape} != scores shape {scores.shape}"
    assert scale_params.shape[0] >= nw, f"scale_params.shape[0]={scale_params.shape[0]} must be >= nw={nw}"
    assert scale_params.shape[1] == num_mix, f"scale_params.shape[1]={scale_params.shape[1]} != num_mix={num_mix}"
    assert out_ufp.shape == scores.shape, f"out_ufp shape {out_ufp.shape} != scores shape {scores.shape}"
    assert out_g.shape == (N1, nw), f"out_g shape {out_g.shape} != (N1={N1}, nw={nw})"
    
    u = responsibilities
    fp = scores
    sbeta = scale_params
    sbeta_h = sbeta[comp_indices, :] # components for this model
    ufp = out_ufp
    g = out_g
    #--------------------------FORTRAN CODE-------------------------
    # for (i = 1, nw) ... for (j = 1, num_mix)
    # ufp(bstrt:bstp) = u(bstrt:bstp) * fp(bstrt:bstp)
    # ufp[bstrt-1:bstp] = u[bstrt-1:bstp] * fp[bstrt-1:bstp]
    # (bstrt:bstp,i) = g(bstrt:bstp,i) + sbeta(j,comp_list(i,h)) * ufp(bstrt:bstp)
    #---------------------------------------------------------------
    # === Subsection: Accumulate Statistics for Parameter Updates ===
    # !--- get g
    # if update_A:

    # einsum is memory-friendly, ~6x faster than naive vectorization on test file
    np.multiply(u, fp, out=ufp)    
    np.einsum('tnj,nj->tn', ufp, sbeta_h, optimize=True, out=g)
    return ufp, g


def accum_updates_and_likelihood(
        config,
        updates,
        state,
        dWtmp,
        LLtmp,
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
    comp_list, comp_used = get_component_indices(config.n_components, config.n_models)

    num_comps = config.n_components
    nw = num_comps
    num_models = config.n_models
    do_reject = config.do_reject
    do_newton = config.do_newton
    newt_start = config.newt_start

    A = state.A
    gm = state.gm
    mu = state.mu
    
    dgm_numer = updates.dgm_numer
    dAK = updates.dAK

    if do_newton:
        dbaralpha_numer = updates.newton.dbaralpha_numer
        dbaralpha_denom = updates.newton.dbaralpha_denom
        dsigma2_numer = updates.newton.dsigma2_numer
        dsigma2_denom = updates.newton.dsigma2_denom
        dkappa_numer = updates.newton.dkappa_numer
        dkappa_denom = updates.newton.dkappa_denom
        dlambda_numer = updates.newton.dlambda_numer
        dlambda_denom = updates.newton.dlambda_denom
    # if update_A:
    # call MPI_REDUCE(dWtmp,dA,nw*nw*num_models,MPI_DOUBLE_PRECISION,MPI_SUM,0,seg_comm,ierr)
    
    dA = dWtmp.copy() # That MPI_REDUCE operation takes dWtmp and accumulates it into dA
    assert dA.shape == dWtmp.shape == (32, 32, 1) == (nw, nw, num_models)
    Wtmp_working = np.zeros((num_comps, num_comps))

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
        # dbaralpha_numer[:, :, :] = dbaralpha_numer_tmp[:, :, :].copy()
        # dbaralpha_denom[:, :, :] = dbaralpha_denom_tmp[:, :, :].copy()
        #---------------------------------------------------------------
        assert dbaralpha_denom[0, 0, 0] == 30504
        # dkappa_numer[:, :, :] = dkappa_numer_tmp[:, :, :].copy()
        # dkappa_denom[:, :, :] = dkappa_denom_tmp[:, :, :].copy()
        # dlambda_numer[:, :, :] = dlambda_numer_tmp[:, :, :].copy()
        # dlambda_denom[:, :, :] = dlambda_denom_tmp[:, :, :].copy()
        # dsigma2_numer[:, :] = dsigma2_numer_tmp[:, :].copy()
        # dsigma2_denom[:, :] = dsigma2_denom_tmp[:, :].copy()
        # NOTE: This is the first newton iteration, and we are already pretty far from the expected values
        # NOTE: The differences are huge.


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
        kappa = np.zeros((num_comps, num_models), dtype=np.float64)
        lambda_ = np.zeros((num_comps, num_models), dtype=np.float64)
        for h, _ in enumerate(range(num_models), start=1):
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
            comp_indices_h = comp_list[:, h_idx] - 1 # shape (nw,)

            # Calculate dkap for all mixtures 
            # dkap = dkappa_numer(j,i,h) / dkappa_denom(j,i,h)
            # kappa(i,h) = kappa(i,h) + baralpha(j,i,h) * dkap
            dkap = dkappa_numer_h / dkappa_denom_h
            # --- Update kappa ---
            # Calculate all update terms and sum along the mixture axis
            kappa_update = np.sum(baralpha_h * dkap, axis=1)
            kappa[:, h_idx] += kappa_update

            # --- Update lambda_ ---
            #--------------------------FORTRAN CODE-------------------------
            # lambda(i,h) = lambda(i,h) + ...
            #       baralpha(j,i,h) * ( dlambda_numer(j,i,h)/dlambda_denom(j,i,h) + dkap * mu(j,comp_list(i,h))**2 )
            #---------------------------------------------------------------
            # mu_selected will have shape (num_mix, nw)
            mu_selected = mu[comp_indices_h, :]

            # Calculate the full lambda update term
            lambda_inner_term = (dlambda_numer_h / dlambda_denom_h) + (dkap * mu_selected**2)
            lambda_update = np.sum(baralpha_h * lambda_inner_term, axis=1)
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
            assert dA.shape == (32, 32, 1) == (nw, nw, num_models)
            dA[:, :, h - 1] *= -1.0 / dgm_numer[h - 1]
            
        np.fill_diagonal(dA[:, :, h_index], dA[:, :, h_index].diagonal() + 1.0)
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
            # on-diagonal elements
            np.fill_diagonal(Wtmp_working, dA[:, :, h - 1].diagonal() / lambda_[:, h - 1])
            # off-diagonal elements
            i_indices, k_indices = np.meshgrid(np.arange(nw), np.arange(num_comps), indexing='ij')
            off_diag_mask = i_indices != k_indices
            assert np.any(off_diag_mask)  # Ensure there are off-diagonal elements
            sk1 = sigma2[i_indices, h-1] * kappa[k_indices, h-1]
            sk2 = sigma2[k_indices, h-1] * kappa[i_indices, h-1]
            positive_mask = (sk1 * sk2 > 0.0)
            if np.any(~positive_mask):
                posdef = False
                no_newt = True
                # This is a placeholder to see if this condition is hit
                assert 1 == 0
            condition_mask = positive_mask & off_diag_mask
            if np.any(condition_mask):
                # # Wtmp(i,k) = (sk1*dA(i,k,h) - dA(k,i,h)) / (sk1*sk2 - dble(1.0))
                numerator = sk1 * dA[i_indices, k_indices, h-1] - dA[k_indices, i_indices, h-1]
                denominator = sk1 * sk2 - 1.0
                Wtmp_working[condition_mask] = (numerator / denominator)[condition_mask]
            # end if (i == k)
            # end do (k)
            # end do (i)
        # end if (do_newton .and. iter >= newt_start)
        elif not do_newton and iter >= newt_start:
            raise NotImplementedError()  # pragma no cover
        if ((not do_newton) or (not posdef) or (iter < newt_start)):
            #  Wtmp = dA(:,:,h)
            assert Wtmp_working.shape == dA[:, :, h - 1].squeeze().shape == (nw, nw)
            Wtmp_working = (dA[:, :, h - 1].squeeze()).copy()
            assert Wtmp_working.shape == (32, 32) == (nw, nw)
        
        #--------------------------FORTRAN CODE-------------------------
        # call DSCAL(nw*nw,dble(0.0),dA(:,:,h),1)
        # call DGEMM('N','N',nw,nw,nw,dble(1.0),A(:,comp_list(:,h)),nw,Wtmp,nw,dble(1.0),dA(:,:,h),nw) 
        #---------------------------------------------------------------
        dA[:, :, h - 1] = 0.0
        dA[:, :, h - 1] += np.dot(A[:, comp_list[:, h - 1] - 1], Wtmp_working)
    # end do (h)

    zeta = np.zeros(num_comps, dtype=np.float64)
    for h, _ in enumerate(range(num_models), start=1):
        h_index = h - 1
        # NOTE: I had an indexing bug in the looped version of this code.
        # But it didn't seem to affect the results.
        
        #--------------------------FORTRAN CODE-------------------------
        # dAk(:,comp_list(i,h)) = dAk(:,comp_list(i,h)) + gm(h)*dA(:,i,h)
        # zeta(comp_list(i,h)) = zeta(comp_list(i,h)) + gm(h)
        #---------------------------------------------------------------
        comp_indices = comp_list[:, h - 1] - 1
        source_columns = gm[h - 1] * dA[:, :, h - 1]
        dAK[comp_indices, :] += source_columns
        zeta[comp_indices] += gm[h - 1]
    
    #--------------------------FORTRAN CODE-------------------------
    # dAk(:,k) = dAk(:,k) / zeta(k)
    # nd(iter,:) = sum(dAk*dAk,1)
    # ndtmpsum = sqrt(sum(nd(iter,:),mask=comp_used) / (nw*count(comp_used)))
    #---------------------------------------------------------------
    dAK[:,:] /= zeta  # Broadcasting division
    # nd is (num_iters, num_comps) in Fortran, but we only store current iteration
    nd = np.sum(dAK * dAK, axis=0)  # Python-only variable name
    assert nd.shape == (num_comps,)

    # comp_used should be 32 length vector of True
    assert isinstance(comp_used, np.ndarray)
    assert comp_used.shape == (num_comps,)
    assert comp_used.dtype == bool
    ndtmpsum = np.sqrt(np.sum(nd) / (nw * np.count_nonzero(comp_used)))
    # end if (update_A)
    
    # if (seg_rank == 0) then
    if do_reject:
        raise NotImplementedError()
        # LL(iter) = LLtmp2 / dble(numgoodsum*nw)
    else:
        # LL(iter) = LLtmp2 / dble(all_blks*nw)
        LLtmp2 = LLtmp  # XXX: In the Fortran code LLtmp2 is the summed LLtmps across processes.
        likelihood = LLtmp2 / (all_blks * nw)
    # TODO: figure out what needs to be returned here (i.e. it is defined in thic func but rest of the program needs it)
    if iter == 1:
        assert dgm_numer[0] == 30504
        # FIXME: this shouldnt be true... dsigma2_number should not be assigned until iter 50
        assert_almost_equal(dsigma2_numer[0, 0], 30517.927488143538, decimal=6)
        assert dsigma2_denom[31, 0] == 30504
        
        assert np.all(dkappa_numer == 0)
        assert np.all(dkappa_denom == 0)
        assert np.all(dlambda_numer == 0)
        assert np.all(dlambda_denom == 0)
        assert np.all(dbaralpha_numer == 0)
        assert np.all(dbaralpha_denom == 0)
        assert_almost_equal(Wtmp_working[0, 0], 0.44757740890010089)
        assert_almost_equal(dA[31, 31, 0], 0.3099478996731922)
        assert_almost_equal(dAK[0, 0], 0.44757153346268763)
        assert_almost_equal(nd[0], 0.20135421232976469)
    elif iter == 2:
        assert dgm_numer[0] == 30504
        assert dsigma2_denom[31, 0] == 30504
        assert_almost_equal(dsigma2_numer[31, 0], 30519.2998249066, decimal=6)              
        assert_almost_equal(dA[31, 31, 0], 0.088792324147082199)
        assert_almost_equal(dAK[0, 0], 0.32313767684058614)

    return (likelihood, ndtmpsum, dAK, no_newt)


def update_params(
        *,
        config,
        state,
        updates,
        metrics,
        wc,
):
    comp_list, _ = get_component_indices(config.n_components, config.n_models)
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

    dgm_numer = updates.dgm_numer
    dalpha_numer = updates.dalpha_numer
    dalpha_denom = updates.dalpha_denom
    dc_numer = updates.dc_numer
    dc_denom = updates.dc_denom
    dmu_numer = updates.dmu_numer
    dmu_denom = updates.dmu_denom
    dbeta_numer = updates.dbeta_numer
    dbeta_denom = updates.dbeta_denom
    drho_numer = updates.drho_numer
    drho_denom = updates.drho_denom
    dAK = updates.dAK

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
        if iter == 1:
            assert dgm_numer == 30504
            assert all_blks == 30504
            assert gm[0] == 1
    # end if (update_gm)

    # if update_alpha:
    # assert alpha.shape == (num_comps, num_mix)
    alpha[:, :] = dalpha_numer / dalpha_denom
    if iter == 1:
        assert_almost_equal(dalpha_numer[0, 0], 8967.4993064961727, decimal=5)
        assert dalpha_denom[0, 0] == 30504
        assert_almost_equal(alpha[0, 0], 0.29397781623708935, decimal=5)

    # if update_c:
    # assert c.shape == (nw, num_models)
    c[:, :] = dc_numer / dc_denom
    if iter == 1:
        assert_almost_equal(dc_numer[0, 0], 0)
        assert dc_denom[0, 0] == 30504
        assert_almost_equal(c[0, 0], 0)
    
    # === Section: Apply Parameter Updates & Rescale ===
    # Apply accumulated statistics to update parameters, then rescale and refresh W/wc.
    # !print *, 'updating A ...'; call flush(6)
    # global lrate, rholrate, lrate0, rholrate0, newtrate, newt_ramp
    if (iter < share_start or (iter % share_iter > 5)):
        if do_newton and (not metrics.no_newt) and (iter >= newt_start):
            if iter == 50:
                assert lrate == .05
                assert rholrate == .05
                assert lrate0 == .05
                assert rholrate0 == .05
                assert newtrate == 1
                assert newt_ramp == 10
                assert_almost_equal(dAK[0, 0], 0.020958681999945186, decimal=4)
                # NOTE: Vectorizing the dorho block loses 1 order of magnitude precision (decimal=5 to decimal=4)
                assert_almost_equal(A[0, 0], 0.96757837792896018, decimal=4)
            # lrate = min( newtrate, lrate + min(dble(1.0)/dble(newt_ramp),lrate) )
            # rholrate = rholrate0
            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            lrate = min(newtrate, lrate + min(1.0 / newt_ramp, lrate))
            rholrate = rholrate0
            A[:, :] -= lrate * dAK
            if iter == 50:
                assert lrate == 0.1
                assert rholrate == 0.05
                # NOTE: Vectorizing the dorho block loses 1 order of magnitude precision (decimal=5 to decimal=4)
                assert_almost_equal(A[0, 0], 0.96548250972896565, decimal=4)
        else:
            if not posdef:
                print("Hessian not positive definite, using natural gradient")
                assert 1 == 0
            
            lrate = min(
                lrate0, lrate + min(1 / newt_ramp, lrate)
                )
            
            rholrate = rholrate0
            if iter == 1:
                assert_almost_equal(lrate0, 0.05)
                assert_almost_equal(lrate, 0.05)
                assert_almost_equal(rholrate0, 0.05)
                assert_almost_equal(rholrate, 0.05)

            # call DAXPY(nw*num_comps,dble(-1.0)*lrate,dAk,1,A,1)
            A[:, :] -= lrate * dAK
            if iter == 1:
                assert_almost_equal(dAK[0, 0], 0.44757153346268763)
                assert_almost_equal(A[0, 0], 0.97750092627907714)
        # end if do_newton
    # end if (update_A)

    # if update_mu:
    mu[:, :] += dmu_numer / dmu_denom
    if iter == 1:
        assert_almost_equal(dmu_numer[0, 0], 6907.8603204569654, decimal=5)
        assert_almost_equal(dmu_denom[0, 0], 22471.172722479747, decimal=3)
        assert_almost_equal(mu[0, 0], -0.69355607517402873)
    
    # if update_beta:
    if iter == 1:
        assert_almost_equal(sbeta[0, 0], 0.96533589542801645)
        assert_almost_equal(dbeta_numer[0, 0], 8967.4993064961727, decimal=5)
        assert_almost_equal(dbeta_denom[0, 0], 10124.98913119294, decimal=5)
        assert_almost_equal(sbetatmp[0, 0], 0.84664104055448097)
        assert invsigmax == 100
        assert invsigmin == 0
    sbeta[:, :] *= np.sqrt(dbeta_numer / dbeta_denom)
    sbetatmp[:, :] = np.minimum(invsigmax, sbeta)
    if iter == 1:
        assert_almost_equal(sbeta[0, 0], 0.90848309104731939)
        assert_almost_equal(sbetatmp[0, 0], 0.90848309104731939)
        assert not sbetatmp[sbetatmp == 100].any()
    sbeta[:, :] = np.maximum(invsigmin, sbetatmp)
    if iter == 1:
        assert_almost_equal(sbeta[0, 0], 0.90848309104731939) # NOTE: this gets updated again below
        assert not sbeta[sbeta == 0].any()
    # end if (update_beta)

    if dorho:
        if iter == 1:
            assert_allclose(rho, 1.5)
        rho[:, :] += (
             rholrate
             * (
                 1.0
                 - (rho / psi(1.0 + 1.0 / rho))
                * drho_numer
                / drho_denom
            )
        )
        rhotmp = np.minimum(maxrho, rho) # shape (num_comps, num_mix)
        assert rhotmp.shape == (config.n_components, config.n_mixtures)
        rho[:, :] = np.maximum(minrho, rhotmp)
        if iter == 1:
            assert maxrho == 2
            assert minrho == 1
            assert_almost_equal(rhotmp[0, 0], 1.4573165687688203)
            assert not rhotmp[rhotmp == maxrho].any()
            assert_almost_equal(rho[0, 0], 1.4573165687688203)
            assert not rho[rho == minrho].any()
        elif iter == 2:
            assert_almost_equal(rhotmp[0, 0], 1.5062036957555023)
            assert not rhotmp[rhotmp == maxrho].any()
            assert not rhotmp[rhotmp == maxrho].any()
    # end if (dorho)

    # !--- rescale
    # !print *, 'rescaling A ...'; call flush(6)
    # from seed import A_FORTRAN
    if doscaling:
        # calculate the L2 norm for each column of A and then use it to normalize that
        # column and scale the corresponding columns in mu and sbeta, but only if the
        # norm is positive.
        # NOTE: this shadows a global variable Anrmk
        Anrmk = np.linalg.norm(A, axis=0)
        positive_mask = Anrmk > 0
        if positive_mask.all():
            A[:, positive_mask] /= Anrmk[positive_mask]
            mu[positive_mask, :] *= Anrmk[positive_mask, None]
            sbeta[positive_mask, :] /= Anrmk[positive_mask, None]
        else:
            raise NotImplementedError()            
    # end if (doscaling)
    if iter == 1:
        # and k == 1
        # assert_almost_equal(A[0, 0], 0.97750092627907714)
        # assert_almost_equal(A[15, 15], 0.984237369637182)
        # assert_almost_equal(A[31, 31], 0.98433353588897787)
        assert_almost_equal(Anrmk[0], 0.98139710406763181, decimal=2) # XXX: watch this value
        # max_rel_error = np.max(np.abs((A - A_FORTRAN) / A_FORTRAN))
        # print(f"Maximum relative error: {max_rel_error:.2e}")  # Should be ~3e-7
        # Since we are vectorizing this wont be true. Id have to go back and get this array
        # From fortran at the 32nd iteration.
        # assert_allclose(A, A_FORTRAN, atol=1e-7)
        # assert max_rel_error < 1e-6  # Very reasonable tolerance
        # assert k == 32  # Just a sanity check for the loop above
        assert_almost_equal(Anrmk[-1], 0.98448954017506363) # XXX: OK here Anrmk matches the Fortran output
        assert_almost_equal(A[15, 15], 0.99984861847601925)
        assert_almost_equal(A[31, 31], 0.99984153789378194)
        assert_almost_equal(sbeta[31, 0], 0.97674982753812623)
        assert_almost_equal(mu[31, 0], -0.8568024781696123)
    elif iter == 2:
        assert_almost_equal(Anrmk[0], 0.9838518400665005) # XXX: Much better!
        # assert k == 32
        assert_almost_equal(Anrmk[-1], 0.99554375802233519)

    if (share_comps and (iter >= share_start) and (iter-share_iter % share_iter == 0)):
        raise NotImplementedError()
    else:
        global free_pass
        free_pass = False
    
    W[:, :, :], wc[:, :] = get_unmixing_matrices(
        iterating=True,
        c=c,
        A=A,
        comp_list=comp_list,
        W=W,
        num_models=num_models,
    )
    if iter == 1:
        assert_almost_equal(W[0, 0, 0], 1.0000820892004447)
        assert_almost_equal(wc[0, 0], 0)

    
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
    assert blocks_in_sample == 30504
    assert num_samples == 1
    assert all_blks == 30504
    dataseg: np.ndarray = raw.get_data() # shape (n_channels, n_times) = (32, 30504)
    assert dataseg.shape == (32, 30504)
    dataseg *= 1e6  # Convert to microvolts
    # Check our value against the Fortran output
    assert_almost_equal(dataseg[0, 0], -35.7974853515625)
    assert_almost_equal(dataseg[1, 0], 2.3078439235687256)
    assert_almost_equal(dataseg[2, 0], -26.776725769042969)

    assert_almost_equal(dataseg[0, 1], -21.326353073120117)
    assert_almost_equal(dataseg[-1, -1], 12.871612548828125)
    assert_almost_equal(dataseg[0, 15252], 28.903553009033203)
    assert_almost_equal(dataseg[0, 29696], -0.48521178960800171)
    assert_almost_equal(dataseg[0, 29700], -2.2563831806182861)
    assert_almost_equal(dataseg[15, 29701], 14.67259693145752)
    assert_almost_equal(dataseg[31, 0], -9.5071401596069336)
    assert_almost_equal(dataseg[20, 29710], -11.413822174072266)

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
