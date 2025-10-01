"""
State management for the AMICA Python port.

This module introduces structured containers to replace globals and clarify
what is persistent model state vs. per-iteration temporaries and aggregated
updates. It keeps allocations explicit and allows reuse of large buffers.

Key containers
- AmicaConfig: immutable configuration and sizes.
    - (nchan, ncomp, nmix, block_size, dtype).
- AmicaState: persistent parameters updated across iterations.
    - W: (n_components, n_components, n_models), A: (n_components, n_components), mu/sbeta/rho: (n_models, n_components), gm: (n_models,).
    - Includes to_dict() for easy serialization.
- AmicaAccumulators: per-iteration aggregated numerators/denominators and grads.
    - (dalpha/dbeta/dmu/drho numerators/denominators, dgm_numer, loglik_sum).
- AmicaMetrics: optional diagnostics for logging/inspection.

Factory helpers
- get_initial_state(cfg, seeds): create an AmicaState from sizes and seeds.

Note: This file is intentionally self-contained and does not depend on the
rest of the code until wired. It is safe to import without side effects.

- Shapes: mu/sbeta/rho use (ncomp, nmix), matching new standard (e.g., mu[comp_indices, :]).
- Workspace is intentionally generic (name → buffer). This lets us wire temps one-by-one without locking the shapes prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
import torch

from amica.constants import rho0

# Removed _alloc helper - using simple np.zeros/np.empty directly like amica.py


@dataclass(slots=True, frozen=True)
class AmicaConfig:
    """Immutable configuration for AMICA."""

    n_features: int  # nchan - number of input channels/features
    n_components: int  # ncomp - number of output components  
    n_models: int  # number of AMICA models
    n_mixtures: int  # nmix - number of mixture components per source
    batch_size: int

    # Execution
    max_iter: int = 200


    # Algorithmic flags
    do_reject: bool = False
    pdftype: int = 0
    do_newton: bool = True
    
    # Tolerances and learning rates
    tol: float = 1e-7
    lrate: float = 0.05
    rholrate: float = 0.05
    newt_start: int = 50
    newtrate: float = 1.0
    newt_ramp: int = 10

    # Numeric
    dtype: torch.dtype = torch.float64


@dataclass(slots=True, repr=False)
class AmicaState:
    """Persistent model parameters/state.

    Arrays follow shapes consistent with the Fortran port:
    - W:  (ncomp, n_feature, nmix)   unmixing matrices per mixture
    - A:  (n_feature, ncomp, nmix)   mixing matrices per mixture (often inv(W))
    - c: (ncomp, n_models)       bias (offset) terms per component and model.
    - mu: (ncomp, nmix)          location parameters per component and mixture
    - sbeta: (ncomp, nmix)       scale parameters per component and mixture
    - rho: (ncomp, nmix)         shape parameters per component and mixture
    - alpha: (ncomp, nmix)       mixing coefficients per component and mixture
    - gm: (n_models,)                mixture weights (prior over models)
    """

    W: NDArray
    A: NDArray
    c: NDArray
    mu: NDArray
    sbeta: NDArray
    rho: NDArray
    alpha: NDArray
    gm: NDArray

    def to_dict(self) -> Dict[str, NDArray]:
        """Return a lightweight serialization of array fields."""
        return {
            "W": self.W,
            "A": self.A,
            "mu": self.mu,
            "sbeta": self.sbeta,
            "rho": self.rho,
            "alpha": self.alpha,
            "gm": self.gm,
            "c": self.c,
        }
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Return a lightweight serialization of array fields as numpy arrays."""
        return {k: v.cpu().numpy() for k, v in self.to_dict().items()}

@dataclass(slots=True, repr=False)
class AmicaAccumulators:
    """Arrays that accumulate across chunks within in one iteration.

    This container mainly helps to shuttle a number of arrays together
    across functions. All these arrays are zeroed at the start of each
    iteration but must persist across chunks within the iteration. If processing the
    dataset in one chunk, these are just calculated once per iteration (thus they are not
    really "accumulators" in that case).

    Shapes:
    - dmu_numer/denom: (ncomp, nmix)
    - dbeta_numer/denom: (ncomp, nmix)
    - dalpha_numer/denom: (ncomp, nmix)
    - drho_numer/denom: (ncomp, nmix)
    - dgm_numer:       (nmix,)
    - dsigma2_numer/denom, dc_numer/denom, etc. can be added as needed.
    """

    dmu_numer: NDArray
    dmu_denom: NDArray

    dbeta_numer: NDArray
    dbeta_denom: NDArray

    dalpha_numer: NDArray
    dalpha_denom: NDArray

    drho_numer: NDArray
    drho_denom: NDArray

    dgm_numer: NDArray

    dc_numer: NDArray
    dc_denom: NDArray

    dA: NDArray
    dAK: NDArray
    loglik_sum: float = 0.0

    newton: Optional[AmicaNewtonAccumulators] = None

    def reset(self) -> None:
        """Zero all per-iteration accumulators in-place.

        Keeps array allocations stable while clearing their contents for the
        next iteration. Extensible: if new fields are added, include them here.
        """
        # Core accumulators
        self.dmu_numer.fill_(0.0)
        self.dmu_denom.fill_(0.0)

        self.dbeta_numer.fill_(0.0)
        self.dbeta_denom.fill_(0.0)

        self.dalpha_numer.fill_(0.0)
        self.dalpha_denom.fill_(0.0)

        self.drho_numer.fill_(0.0)
        self.drho_denom.fill_(0.0)

        self.dgm_numer.fill_(0.0)

        self.dc_numer.fill_(0.0)
        self.dc_denom.fill_(0.0)

        self.dA.fill_(0.0)
        self.dAK.fill_(0.0)
        self.loglik_sum = 0.0

        # If Newton accumulators are present, zero them as well
        if self.newton is not None:
            self.newton.dbaralpha_numer.fill_(0.0)
            self.newton.dbaralpha_denom.fill_(0.0)
            self.newton.dkappa_numer.fill_(0.0)
            self.newton.dkappa_denom.fill_(0.0)
            self.newton.dlambda_numer.fill_(0.0)
            self.newton.dlambda_denom.fill_(0.0)
            self.newton.dsigma2_numer.fill_(0.0)
            self.newton.dsigma2_denom.fill_(0.0)

@dataclass(slots=True, repr=False)
class AmicaNewtonAccumulators:
    """Additional accumulators for Newton updates.

    Shapes:
    - dbaralpha_numer/denom: (n_components, n_mixtures, n_models)
    - dkappa_numer/denom: (n_components, n_mixtures, n_models)
    - dlambda_numer/denom: (n_components, n_mixtures, n_models)
    - dsigma2_numer/denom: (n_components, n_mixtures, n_models)
    """

    dbaralpha_numer: NDArray
    dbaralpha_denom: NDArray

    dkappa_numer: NDArray
    dkappa_denom: NDArray

    dlambda_numer: NDArray
    dlambda_denom: NDArray

    dsigma2_numer: NDArray
    dsigma2_denom: NDArray

@dataclass(slots=True)
class IterationMetrics:
    """Minimal per-iteration diagnostics.

    This container tracks iteration specific metadata that evolves during training and
    influences convergence behavior, but are not learnable parameters themselves.
    The container helps oraganize the values and pass them around functions.
    
    Fields:
    - ndtmpsum: Total norm of the weight gradient, summed across all components; computed as sqrt(mean of per-component squared update norms).
    """

    iter: int                           # 1-based iteration index
    loglik: Optional[float] = None      # total log-likelihood for the iteration
    ndtmpsum: Optional[float] = None    # normalized update norm for the iteration
    lrate: Optional[float] = None       # learning rate used for the iteration
    rholrate: Optional[float] = None    # rho learning rate used for the iteration
    lrate0: Optional[float] = None      # initial learning rate (from config)
    rholrate0: Optional[float] = None   # initial rho learning rate (from config)
    newtrate: Optional[float] = None    # Newton learning rate used for the iteration
    ll_inc: float = 0.0                 # improvement vs previous iteration
    step_time_s: Optional[float] = None
    numincs: Optional[int] = None
    numdecs: Optional[int] = None


@dataclass(slots=True)
class AmicaHistory:
    """Append-only container of IterationMetrics across training.

    Provides lightweight accessors commonly used by callers and tests.
    """

    metrics: List[IterationMetrics] = field(default_factory=list)

    def append(self, m: IterationMetrics) -> None:
        self.metrics.append(m)

    # Convenience accessors (kept minimal)
    def loglik_array(self) -> torch.Tensor:
        return torch.tensor([m.loglik for m in self.metrics], dtype=torch.float64)

    def ll_inc_array(self) -> torch.Tensor:
        return torch.tensor([m.ll_inc for m in self.metrics], dtype=torch.float64)

    def last(self) -> Optional[IterationMetrics]:
        return self.metrics[-1] if self.metrics else None

# TODO: consider making an IterationContext or IterationMetrics class
# to bundle ephemeral constants that shuttle together across functions.
# e.g. Dsum, sldet, comp_list etc.
# - sldet: float               sum log det(W) across components/models.

# TODO: consider making this a class method of AmicaState
def get_initial_state(
    cfg: AmicaConfig,
    *,
    seeds: Optional[Mapping[str, NDArray]] = None,
) -> AmicaState:
    """Create an initial AmicaState.

    Initialize arrays following the same patterns as amica.py.
    If `seeds` provides 'W', 'sbeta', or 'mu', they are used to initialize
    the corresponding fields.
    """
    num_comps = cfg.n_components  # Match amica.py variable names
    num_models = cfg.n_models  
    num_mix = cfg.n_mixtures
    dtype = cfg.dtype

    seeds = {} if seeds is None else dict(seeds)

    # W - match amica.py: W = np.zeros((num_comps, num_comps, num_models))
    if "W" in seeds:
        W = torch.tensor(seeds["W"], dtype=dtype)
        assert W.shape == (num_comps, num_comps, num_models), (
            f"W seed shape {W.shape} != {(num_comps, num_comps, num_models)}"
        )
    else:
        W = torch.empty((num_comps, num_comps, num_models), dtype=dtype)  # Weights for each model
    assert W.dtype == torch.float64

    # A - match amica.py: A = np.zeros((num_comps, num_comps))  
    A = torch.zeros((num_comps, num_comps), dtype=dtype)

    c = torch.zeros((num_comps, num_models), dtype=dtype)  # Bias terms per component and model
    # sbeta, mu, rho - match amica.py patterns
    if "sbeta" in seeds:
        sbeta = torch.tensor(seeds["sbeta"], dtype=dtype)
        assert sbeta.shape == (num_comps, num_mix)
    else:
        sbeta = torch.empty((num_comps, num_mix), dtype=dtype)

    if "mu" in seeds:
        mu = torch.tensor(seeds["mu"], dtype=dtype)
        assert mu.shape == (num_comps, num_mix)
    else:
        mu = torch.zeros((num_comps, num_mix), dtype=dtype)

    rho = torch.full(fill_value=rho0, size=(num_comps, num_mix), dtype=dtype)  # Shape parameters
    
    # Initialize alpha (mixing coefficients) to zeros - will be computed in first iteration
    alpha = torch.zeros((num_comps, num_mix), dtype=dtype)

    gm = torch.full(fill_value=1.0 / num_models, size=(num_models,), dtype=dtype)  # Equal weights initially
    return AmicaState(W=W, A=A, c=c, mu=mu, sbeta=sbeta, rho=rho, alpha=alpha, gm=gm)


# TODO: consider making this a class method of AmicaAccumulators
def initialize_accumulators(cfg: AmicaConfig) -> AmicaAccumulators:
    """Allocate zeroed update accumulators with shapes from the config."""
    num_comps = cfg.n_components
    num_models = cfg.n_models  
    num_mix = cfg.n_mixtures
    dtype = cfg.dtype
    do_newton = cfg.do_newton
    shape_2 = (num_comps, num_mix)

    # Match amica.py initialization patterns
    dgm_numer = torch.zeros(num_models, dtype=dtype)

    # Update accumulators - standardized: (num_comps, num_mix) shape
    dmu_numer = torch.zeros(shape_2, dtype=dtype)
    dmu_denom = torch.zeros(shape_2, dtype=dtype)

    dbeta_numer = torch.zeros(shape_2, dtype=dtype)
    dbeta_denom = torch.zeros(shape_2, dtype=dtype)

    dalpha_numer = torch.zeros(shape_2, dtype=dtype)
    dalpha_denom = torch.zeros(shape_2, dtype=dtype)

    drho_numer = torch.zeros(shape_2, dtype=dtype)
    drho_denom = torch.zeros(shape_2, dtype=dtype)

    dc_numer = torch.zeros((num_comps, num_models), dtype=dtype)
    dc_denom = torch.zeros((num_comps, num_models), dtype=dtype)

    dA = torch.zeros((num_comps, num_comps, num_models), dtype=dtype)  # Derivative of A per model
    dAK = torch.zeros((num_comps, num_comps), dtype=dtype)  # Derivative of A

    if do_newton:
        # NOTE: Amica authors gave newton arrays 3 dims, but gradient descent 2 dims
        shape_3 = (num_comps, num_mix, num_models)

        dbaralpha_numer = torch.zeros(shape_3, dtype=dtype)
        dbaralpha_denom = torch.zeros(shape_3, dtype=dtype)

        dkappa_numer = torch.zeros(shape_3, dtype=dtype)
        dkappa_denom = torch.zeros(shape_3, dtype=dtype)

        dlambda_numer = torch.zeros(shape_3, dtype=dtype)
        dlambda_denom = torch.zeros(shape_3, dtype=dtype)

        # These are 2D in the Fortran code, which actually uses nw x num_models
        dsigma2_numer = torch.zeros((num_comps, num_models), dtype=dtype)
        dsigma2_denom = torch.zeros((num_comps, num_models), dtype=dtype)

        newton = AmicaNewtonAccumulators(
            dbaralpha_numer=dbaralpha_numer,
            dbaralpha_denom=dbaralpha_denom,
            dkappa_numer=dkappa_numer,
            dkappa_denom=dkappa_denom,
            dlambda_numer=dlambda_numer,
            dlambda_denom=dlambda_denom,
            dsigma2_numer=dsigma2_numer,
            dsigma2_denom=dsigma2_denom,
        )
    else:
        newton = None

    return AmicaAccumulators(
        dgm_numer=dgm_numer,
        dmu_numer=dmu_numer,
        dmu_denom=dmu_denom,
        dalpha_numer=dalpha_numer,
        dalpha_denom=dalpha_denom,
        dbeta_numer=dbeta_numer,
        dbeta_denom=dbeta_denom,
        drho_numer=drho_numer,
        drho_denom=drho_denom,
        dc_numer=dc_numer,
        dc_denom=dc_denom,
        dA=dA,
        dAK=dAK,
        loglik_sum=0.0,
        newton=newton,
    )


def reset_accumulators(u: AmicaAccumulators) -> None:
    """Zero all per-iteration accumulators in-place.

    This avoids reallocations by reusing the same AmicaAccumulators instance across
    iterations. It resets numerators/denominators, the weight gradients, the
    mixture numerators, and Newton accumulators if present.
    """
    # Core accumulators

    u.dmu_numer.fill_(0.0)
    u.dmu_denom.fill_(0.0)

    u.dbeta_numer.fill_(0.0)
    u.dbeta_denom.fill_(0.0)

    u.dalpha_numer.fill_(0.0)
    u.dalpha_denom.fill_(0.0)

    u.drho_numer.fill_(0.0)
    u.drho_denom.fill_(0.0)

    u.dgm_numer.fill_(0.0)

    # Scalar diagnostics
    u.loglik_sum = 0.0

    # Optional Newton accumulators
    if u.newton is not None:
        n = u.newton
        n.dbaralpha_numer.fill_(0.0)
        n.dbaralpha_denom.fill_(0.0)
        n.dkappa_numer.fill_(0.0)
        n.dkappa_denom.fill_(0.0)
        n.dlambda_numer.fill_(0.0)
        n.dlambda_denom.fill_(0.0)
        n.dsigma2_numer.fill_(0.0)
        n.dsigma2_denom.fill_(0.0)


__all__ = [
    "AmicaConfig",
    "AmicaState",
    "AmicaAccumulators",
    "AmicaMetrics",
    "get_initial_state",
    "initialize_accumulators",
    "reset_accumulators",
]
