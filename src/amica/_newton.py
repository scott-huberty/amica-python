"""Quasi-Newton related computations for estimation of gradient curvature terms."""
import torch


def compute_newton_terms(*, config, accumulators, mu):
    """Compute second-order curvature or scaling terms for Hessian approximation.

    Parameters
    ----------
    config : amica.state.Config
        Configuration object containing model parameters. Specifically, it should have
        ``n_components`` and ``dtype`` attributes.
    accumulators : amica.state.Accumulators
        Accumulators instance containing intermediate tensors needed for Newton term
        calculations. It should have tensors stored in the following attributes:

        - ``dbaralpha_numer`` (n_components, n_mixtures)
        - ``dbaralpha_denom`` (n_components, n_mixtures)
        - ``dsigma2_numer`` (n_components,)
        - ``dsigma2_denom`` (n_components,)
        - ``dkappa_numer`` (n_components, n_mixtures)
        - ``dkappa_denom`` (n_components, n_mixtures)
        - ``dlambda_numer`` (n_components, n_mixtures)
        - ``dlambda_denom`` (n_components, n_mixtures)

    mu : torch.Tensor
        Tensor of shape (n_components, n_mixtures) representing the mean estimates for
        each component and mixture. This is typically stored in the state object
        (``state.mu``).

    Returns
    -------
    dict
        A dictionary containing the computed Newton terms:
        - ``baralpha``: Tensor of shape (n_components, n_mixtures)
        - ``sigma2``: Tensor of shape (n_components,)
        - ``kappa``: Tensor of shape (n_components,)
        - ``lambda_``: Tensor of shape (n_components,)
    """
    #--------------------------FORTRAN CODE--------------------------------------------
    # baralpha = dbaralpha_numer / dbaralpha_denom
    # sigma2 = dsigma2_numer / dsigma2_denom
    # kappa = dble(0.0)
    # lambda = dble(0.0)
    #-----------------------------------------------------------------------------------
    # weighting factor: how much mixture j contributes to source i
    baralpha = accumulators.newton.dbaralpha_numer / accumulators.newton.dbaralpha_denom
    # variance estimate for source i
    sigma2 = accumulators.newton.dsigma2_numer / accumulators.newton.dsigma2_denom
    # curvature terms
    kappa = torch.zeros(
        (config.n_components,), dtype=config.dtype, device=config.device
        )
    lambda_ = torch.zeros(
        (config.n_components,), dtype=config.dtype, device=config.device
        )

    # Calculate dkap for all mixtures
    dkap = accumulators.newton.dkappa_numer / accumulators.newton.dkappa_denom
    kappa += torch.sum(baralpha * dkap, dim=1)

    #--------------------------FORTRAN CODE-------------------------
    # lambda(i,h) = lambda(i,h) + ...
    #       baralpha(j,i,h) * ( dlambda_numer(j,i,h)/dlambda_denom(j,i,h) + ...
    #---------------------------------------------------------------
    lambda_inner_term = (
        (accumulators.newton.dlambda_numer / accumulators.newton.dlambda_denom)
        + (dkap * mu**2)
    )
    lambda_ += torch.sum(baralpha * lambda_inner_term, dim=1)

    return {
        "baralpha": baralpha,       # (n_components, n_mixtures)
        "sigma2": sigma2,           # (n_components,)
        "kappa": kappa,             # (n_components,)
        "lambda_": lambda_,         # (n_components,)
    }
