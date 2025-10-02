"""Whitening, unmixing helpers, determinant computation, etc."""

import torch
import numpy as np
from typing import Literal, Optional, Tuple
from amica._types import ComponentsVector, DataArray2D, WeightsArray


def get_unmixing_matrices(
        *,
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
    - In the Fortran code, the variable Ptmp(bstrt:bstp,h) holds the initial
    model log-likelihood for model h across the data block (bstrt:bstp). This gets
    copied into modloglik. 
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


def pre_whiten(
        *,
        X: DataArray2D,
        n_components: Optional[int] = None,
        mineig: float = 1e-6,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_approx_sphere: bool = True,
        sphering_matrix: Optional[WeightsArray] = None,
        inplace: bool = True,
) -> Tuple[DataArray2D, WeightsArray, float, WeightsArray, ComponentsVector | None]:
    """
    Pre-whiten the input data matrix X prior to ICA.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
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
        If True, use approximate sphering.
    inplace : bool
        If True, modify X in place. If False, make a copy of X and modify that.
    
    Returns
    -------
    X : array, shape (n_samples, n_features)
        The whitened data matrix. This is a copy of the input data if inplace is False,
        otherwise it is the mutated input data itself.
    whitening_matrix : array, shape (n_features, n_features)
        The whitening/sphering matrix applied to the data. If do_sphere is False, then
        this is the variance normalization matrix.
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
    assert dataseg.ndim == 2, f"X must be 2D, got {dataseg.ndim}D"
    # !---------------------------- get the mean --------------------------------
    n_samples, nx = dataseg.shape
    if n_components is None:
        n_components = nx
    
    # ---- Mean-centering ----
    if do_mean:
        print("getting the mean ...")
        mean = dataseg.mean(axis=0)
        # !--- subtract the mean
        dataseg -= mean[None, :]  # Subtract mean from each channel

    # ---- Covariance ----
    print(" Getting the covariance matrix ...")
    # Compute the covariance matrix
    # The Fortran code only computes the upper triangular part of the covariance matrix

    # -------------------- FORTRAN CODE ---------------------------------------
    # call DSCAL(nx*nx,dble(0.0),Stmp,1)
    # call DSYRK('L','N',nx,blk_size(seg),dble(1.0),dataseg(seg)%data(:,bstrt:bstp)...
    # call DSCAL(nx*nx,dble(1.0)/dble(cnt),S,1)
    #------------------------------------------------------------------------
    Cov = dataseg.T @ dataseg / n_samples

    # ---- Eigen-decomposition
    print(f"doing eig nx = {nx}")
    eigvals, eigvecs = np.linalg.eigh(Cov) # ascending order

    min_eigs = eigvals[:min(nx//2, 3)]
    max_eigs = eigvals[::-1][:3]
    print(f"minimum eigenvalues: {min_eigs}")
    print(f"maximum eigenvalues: {max_eigs}")
    
    # keep only valid eigs (pcakeep)
    # Do we need to pass numeigs to optimize if sum(eigvals > mineig) < n_components?
    numeigs = min(n_components, sum(eigvals > mineig)) # np.linalg.matrix_rank?
    print(f"num eigvals kept: {numeigs}")

    # Log determinant of the whitening matrix
    if numeigs == nx:
        sldet = -0.5 * np.sum(np.log(eigvals))
    else:
        sldet = -0.5 * np.sum(np.log(eigvals[::-1][:numeigs]))
    
    # 1) reorder eigenvectors (descending eigenvalues)
    order = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[order]
    # UNCSCALED eigenvectors
    Stmp = eigvecs[:, order].T.copy()
    Stmp2 = Stmp.copy()
    # 2) SCALED eigenvectors
    Stmp2[:numeigs, :] /= np.sqrt(eigvals_desc[:numeigs, None])

    # ---- Sphere or variance normalize ----
    if do_sphere:
        print("Sphering the data...")
        if numeigs == nx:
            # call DSCAL(nx*nx,dble(0.0),S,1)
            if do_approx_sphere:
                # Zero-copy assignment
                S = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            else:
                # call DCOPY(nx*nx,Stmp2,1,S,1)
                S = Stmp2.copy()
        else:
            if do_approx_sphere:
                # -------------------- FORTRAN CODE -------------------------------------
                # call DSCAL(nx*blk_size(seg),dble(0.0),xtmp(:,1:blk_size(seg)),1)
                # call DGEMM('N','N',nx,blk_size(seg),nx,dble(1.0),S,nx,dataseg(seg)%data
                # call DCOPY(nx*blk_size(seg),xtmp(:,1:blk_size(seg)),1,dataseg(seg)
                # ------------------------------------------------------------------------

                # This is a direct translation of the Fortran code
                # Which was hard to follow...
                # I think this is ZCA whitening
                
                # 3) Unscaled eigenvectors goes into SVD
                S = Stmp.copy()

                # 4) SVD on leading block of Unscaled eigenvectors
                U, s, VT = np.linalg.svd(S[:numeigs, :numeigs], full_matrices=True)
                Stmp[:numeigs, :numeigs] = VT.T
                S[:numeigs, :numeigs] = U.T
                
                # 5) Stmp3 = Stmp^T @ S^T (numeigs×numeigs block)
                Stmp3 = Stmp[:numeigs, :numeigs] @ S[:numeigs, :numeigs]
                
                # 6) zero S and form final S = Stmp3 @ Stmp2
                S.fill(0.0)
                S[:numeigs, :] = Stmp3 @ Stmp2[:numeigs, :]
            else:
                # I think this is PCA whitening
                S = Stmp2.copy()
            # raise NotImplementedError()
    else:
        # !--- just normalize by the channel variances (don't sphere)
        # -------------------- FORTRAN CODE ---------------------------------------
        # call DCOPY(nx*nx,S,1,Stmp,1)
        # call DSCAL(nx*nx,dble(0.0),S,1)
        #------------------------------------------------------------------------
        S = np.zeros_like(Cov) # This is S in Fortran code
        # Zero out the lower triangle to have parity with Fortran
        sldet = 0.0
        for i in range(nx):
            if np.triu(Cov)[i, i] > 0:
                S[i, i] = 1.0 / np.sqrt(Cov[i, i])
                sldet += 0.5 * np.log(S[i, i])
            numeigs = nx
    # -------------------- FORTRAN CODE ---------------------------------------
    # call DSCAL(nx*blk_size(seg),dble(0.0),xtmp(:,1:blk_size(seg)),1)
    # call DGEMM('N','N',nx,blk_size(seg),nx,dble(1.0),S,nx,dataseg(seg)%data(:,bstrt:bstp),nx,dble(1.0),xtmp(:,1:blk_size(seg)),nx)
    # call DCOPY(nx*blk_size(seg),xtmp(:,1:blk_size(seg)),1,dataseg(seg)%data(:,bstrt:bstp),1)
    # -------------------------------------------------------------------------
    dataseg = np.matmul(dataseg, S.T, out=dataseg)  # In-place if possible

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
    assert dataseg.shape == (n_samples, nx), f"dataseg shape {dataseg.shape} != (n_samples, n_features) = ({n_samples}, {nx})"
    return dataseg, S, sldet, Winv, mean