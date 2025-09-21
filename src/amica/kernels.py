import torch

from typing import Optional, Tuple, Literal

from amica._types import (
    DataArray2D,
    WeightsArray,
    ComponentsVector,
    SourceArray2D,
    SourceArray3D,
    ParamsArray,
    ParamsModelArray,
    SamplesVector,
    LikelihoodArray,
)

from amica.constants import LOG_2, LOG_SQRT_PI

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
    X : array, shape (n_samples, n_features)
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
    assert dataseg.shape[1] == W.shape[1], (f"X n_features {dataseg.shape[1]} != W n_features {W.shape[1]}")

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
        b = torch.matmul(dataseg, W.T)
    # end else
    # Subtract the weight correction factor
    b -= wc
    return b


def compute_source_densities(
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
    nw = b.shape[1]
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
        if torch.any(~torch.isfinite(log_mixture_weights)):
            raise RuntimeError("Non-finite log mixture weights encountered.")
        if torch.any(~torch.isfinite(log_scales)):
            raise RuntimeError("Non-finite log scales encountered.")

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
        Output array for per-sample log-likelihood for this model, mutated in-place.
        If None, a new array is allocated, but note that AMICA expects this array
        to be pre-filled with an initial value. See get_initial_model_log_likelihood.
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
        z = log_densities.clone()
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
        *, modloglik: LikelihoodArray, out: Optional[LikelihoodArray] = None
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
    if out is not None:
        assert out.size() == modloglik.size(), (
            f"out shape {out.shape} != modloglik shape {modloglik.shape}"
        )
        v = out
    else:
        v = torch.empty_like(modloglik)
    #--------------------------FORTRAN CODE-------------------------
    # v(bstrt:bstp,h) = dble(1.0) / exp(P(bstrt:bstp) - Ptmp(bstrt:bstp,h))
    #---------------------------------------------------------------

    num_models = modloglik.shape[1]
    # fast-path: if only one model, skip softmax and set responsibilities to 1
    if num_models == 1:
        v.fill_(1.0)
    else:
        v = torch.softmax(modloglik, dim=-1, out=out) # across models
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
        # This is usually a small loop, and ensures we get a view of the arrays
        if lap_mask.any():
            for i, j in zip(*lap_mask.nonzero(as_tuple=True)):
                out_scores[:, i, j] = torch.sign(y[:, i, j])
        if gau_mask.any():
            for i, j in zip(*gau_mask.nonzero(as_tuple=True)):
                out_scores[:, i, j] = torch.multiply(y[:, i, j], 2.0)
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


def precompute_weighted_scores(
        *,
        scores: SourceArray3D,
        weighted_responsibilities: SourceArray3D,
        out_ufp: Optional[SourceArray3D] = None,
) -> SourceArray3D:
    """
    Compute the weighted score function and g update for the current model.

    This is just the element-wise multiplication of the score function (fp) with the
    weighted responsibilities (u). AMICA pre-computes this because it is used about 5
    times in the M-step (i.e. it avoids recomputing the same thing multiple times).

    Parameters
    ----------
    scores : torch.Tensor
        The score function (fp) of shape (n_samples, n_components, n_mixtures).
        Not modified.
    weighted_responsibilities : torch.Tensor
        The responsibilities (u) of shape (n_samples, n_components, n_mixtures).
        Not modified.
    out_ufp: torch.Tensor or None
        Optional output buffer to write the weighted scores into. If None, a new
        array is allocated. If provided, must be of shape (n_samples, n_components,
        n_mixtures), and this array will be modified in-place and returned. E.g. AMICA
        uses the `fp` array as workspace for this when newton updates are not done
        (e.g. do_newton=False, or do_newton=True but iter < newt_start), because ufp
        and fp lifetimes only overlap when newton updates are done.

    Returns
    -------
    out_ufp : torch.Tensor
        The score function weighted by model-weighted mixture-responsibilities.
        shape (n_samples, n_components, n_mixtures), modified in place.
    """
    u = weighted_responsibilities
    fp = scores
    assert u.shape == fp.shape, (
        f"responsibilities shape {u.shape} != scores shape {fp.shape}"
    )
    if out_ufp is not None:
        assert out_ufp.shape == fp.shape, (
            f"out_ufp shape {out_ufp.shape} != scores shape {fp.shape}"
        )
    #--------------------------FORTRAN CODE-------------------------
    # ufp(bstrt:bstp) = u(bstrt:bstp) * fp(bstrt:bstp)
    #---------------------------------------------------------------
    ufp = torch.multiply(u, fp, out=out_ufp)
    return ufp


def compute_scaled_scores(
        *, weighted_scores: SourceArray3D, scales: ParamsArray
        ) -> SourceArray2D:
    """
    Weigh ufp by the per-component mixture scale parameters (sbeta).
    
    Parameters
    ----------
    weighted_scores : torch.Tensor
        The mixture‑responsibility–weighted scores (ufp)
        shape (n_samples, n_components, n_mixtures). Not modified.
    scales : torch.Tensor
        Scale parameters (sbeta) of shape (n_components, n_mixtures). Not modified.
    
    Returns
    -------
    torch.Tensor
        The scaled scores (g) of shape (n_samples, n_components), i.e. ufp further
        weighted by the per-component summed-mixture scale parameters.
    
    Notes
    -----
    Fortran reference:
        call DSCAL(nw*tblksize,dble(0.0),g(bstrt:bstp,:),1)
        g(bstrt:bstp,i) = g(bstrt:bstp,i) + sbeta(j,comp_list(i,h)) * ufp(bstrt:bstp)
    """
    sbeta = scales
    ufp = weighted_scores
    assert ufp.ndim == 3, f"ufp must be 3D, got {ufp.ndim}D"
    assert sbeta.ndim == 2, f"sbeta must be 2D, got {sbeta.ndim}D"
    assert ufp[0,:,:].squeeze().shape == sbeta.shape, (
        f"ufp shape {ufp.shape} and sbeta shape {sbeta.shape} are incompatible"
    )
    g = torch.sum(ufp * sbeta, dim=-1)
    return g



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
    assert X.shape[0] == v.shape[0], (
        f"X n_samples {X.shape[0]} != model responsibilities length {v.shape[0]}"
    )
    assert vsum.numel() == 1, f"vsum must be a scalar, got {vsum}"
    assert out_numer.shape == (X.shape[1],), (
        f"out_numer shape {out_numer.shape} != (n_components,) "
    )
    assert out_denom.shape == (X.shape[1],), (
        f"out_denom shape {out_denom.shape} != (n_components,) "
        f"= ({X.shape[1]},)"
    )
    if do_reject:
        raise NotImplementedError()
        # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,dataseg(seg)%goodinds(xstr
    else:
        #--------------------------FORTRAN CODE-------------------------
        # tmpsum = sum( v(bstrt:bstp,h) * dataseg(seg)%data(i,xstrt:xstp) )
        #---------------------------------------------------------------
        tmpsum_c_vec = dataseg.T @ v  # Shape: (n_components,)
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
    assert isinstance(epsdble, float), f"epsdble must be a float, got {epsdble}"
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