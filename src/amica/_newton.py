"""Quasi-Newton related computations for estimation of gradient curvature terms."""
import torch

from ._batching import get_component_slice


def compute_newton_terms(*, config, accumulators, mu):
    """Compute second-order curvature or scaling terms for Hessian approximation.
    
    Parameters
    ----------
    config : amica.state.Config
        Configuration object containing model parameters. Specifically, it should have
        ``n_components``, ``n_models``, and ``dtype`` attributes.
    accumulators : amica.state.Accumulators
        Accumulators instance containing intermediate tensors needed for Newton term
        calculations. It should have tensors stored in the following attributes:

        - ``dbaralpha_numer`` (n_components, n_mixtures, n_models)
        - ``dbaralpha_denom`` (n_components, n_mixtures, n_models)
        - ``dsigma2_numer`` (n_components, n_models)
        - ``dsigma2_denom`` (n_components, n_models)
        - ``dkappa_numer`` (n_components, n_models)
        - ``dkappa_denom`` (n_components, n_models)
        - ``dlambda_numer`` (n_components, n_models)
        - ``dlambda_denom`` (n_components, n_models)

    mu : torch.Tensor
        Tensor of shape (n_components, n_mixtures) representing the mean estimates for
        each component and mixture. This is typically stored in the state object
        (``state.mu``).

    Returns
    -------
    dict
        A dictionary containing the computed Newton terms:
        - ``baralpha``: Tensor of shape (n_components, n_mixtures, n_models)
        - ``sigma2``: Tensor of shape (n_components, n_models)
        - ``kappa``: Tensor of shape (n_components, n_models)
        - ``lambda_``: Tensor of shape (n_components, n_models)
    """
    #--------------------------FORTRAN CODE--------------------------------------------
    # baralpha = dbaralpha_numer / dbaralpha_denom
    # sigma2 = dsigma2_numer / dsigma2_denom
    # kappa = dble(0.0)
    # lambda = dble(0.0)
    #-----------------------------------------------------------------------------------
    # weighting factor: how much mixture j contributes to the source i in model h
    baralpha = accumulators.newton.dbaralpha_numer / accumulators.newton.dbaralpha_denom
    # variance estimate for source i in model h
    sigma2 = accumulators.newton.dsigma2_numer / accumulators.newton.dsigma2_denom
    # curvature terms
    kappa = torch.zeros((config.n_components, config.n_models), dtype=config.dtype)
    lambda_ = torch.zeros((config.n_components, config.n_models), dtype=config.dtype)

    # Compute Kappa and Lambda curvature terms
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
        mu_selected = mu[comp_slice, :]
        # Calculate the full lambda update term
        lambda_inner_term = (
            (dlambda_numer_h / dlambda_denom_h) + (dkap * mu_selected**2)
            )
        lambda_update = torch.sum(baralpha_h * lambda_inner_term, dim=1)
        lambda_[:, h_idx] += lambda_update
        # end do (j)
        # end do (i)
    # end do (h)
    return {
        "baralpha": baralpha,       # (n_components, n_mixtures, n_models)
        "sigma2": sigma2,           # (n_components, n_models)
        "kappa": kappa,             # (n_components, n_models)
        "lambda_": lambda_,         # (n_components, n_models)
    }
