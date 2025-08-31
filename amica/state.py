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
- AmicaWorkspace: reusable temporary buffer registry (allocated on demand).
    - get(name, shape, dtype=None, init='empty'|'zeros'|'ones') allocates lazily and reuses.
- AmicaUpdates: per-iteration aggregated numerators/denominators and grads.
    - (dalpha/dbeta/dmu/drho numerators/denominators, dgm_numer, loglik_sum).
- AmicaMetrics: optional diagnostics for logging/inspection.

Factory helpers
- get_initial_state(cfg, seeds): create an AmicaState from sizes and seeds.
- get_workspace(cfg, block_size): create a workspace with on-demand buffers.

Note: This file is intentionally self-contained and does not depend on the
rest of the code until wired. It is safe to import without side effects.

- Shapes: mu/sbeta/rho use (nmix, ncomp), matching your current usage (e.g., mu[:, comp_indices]).
- Workspace is intentionally generic (name → buffer). This lets us wire temps one-by-one without locking the shapes prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from constants import rho0

# Removed _alloc helper - using simple np.zeros/np.empty directly like amica.py


@dataclass(slots=True, frozen=True)
class AmicaConfig:
    """Immutable configuration for AMICA."""

    n_features: int  # nchan - number of input channels/features
    n_components: int  # ncomp - number of output components  
    n_models: int  # number of AMICA models
    n_mixtures: int  # nmix - number of mixture components per source

    # Execution
    max_iter: int = 200
    block_size: int = 512

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
    dtype: np.dtype = np.float64


@dataclass(slots=True, repr=False)
class AmicaState:
    """Persistent model parameters/state.

    Arrays follow shapes consistent with the Fortran port:
    - W:  (ncomp, nchan, nmix)   unmixing matrices per mixture
    - A:  (nchan, ncomp, nmix)   mixing matrices per mixture (often inv(W))
    - mu: (nmix, ncomp)          location parameters per mixture and component
    - sbeta: (nmix, ncomp)       scale parameters per mixture and component
    - rho: (nmix, ncomp)         shape parameters per mixture and component
    - alpha: (nmix, ncomp)       mixing coefficients per mixture and component
    - gm: (nmix,)                mixture weights (prior over models)
    - sldet: float               sum log det(W) across components/models.
    """

    W: NDArray
    A: NDArray
    mu: NDArray
    sbeta: NDArray
    rho: NDArray
    alpha: NDArray
    gm: NDArray
    sldet: float = 0.0

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
        }


@dataclass
class AmicaWorkspace:
    """Reusable temporary buffers.

    Use `get` to request a named buffer with the desired shape. The same
    buffer name should always be requested with a consistent shape/dtype.
    If a name is requested with a different shape/dtype, the buffer is
    reallocated to match the request.
    """

    block_size: int
    dtype: np.dtype
    _buffers: Dict[str, NDArray] = field(default_factory=dict)

    def get(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        dtype: Optional[np.dtype] = None,
        init: str = "empty",
    ) -> NDArray:
        """Get or create a buffer by name.

        - name: unique identifier, e.g. 'z0', 'Ptmp', 'tmpvec'.
        - shape: requested shape. If existing buffer differs, it is reallocated.
        - dtype: if None, defaults to workspace dtype.
        - init: 'empty'|'zeros'|'ones' controls initialization when allocating.
        """
        dtype = self.dtype if dtype is None else dtype
        arr = self._buffers.get(name)
        if arr is None or arr.shape != shape or arr.dtype != dtype:
            if init == "zeros":
                arr = np.zeros(shape, dtype=dtype)
            elif init == "ones":
                arr = np.ones(shape, dtype=dtype)
            else:
                arr = np.empty(shape, dtype=dtype)  # Simple empty allocation
            self._buffers[name] = arr
        return arr

    def clear(self, names: Optional[Iterable[str]] = None) -> None:
        """Drop one or more buffers. If names is None, clear all buffers."""
        if names is None:
            self._buffers.clear()
        else:
            for n in names:
                self._buffers.pop(n, None)


@dataclass(slots=True, repr=False)
class AmicaUpdates:
    """Aggregated updates computed in one iteration.

    Shapes:
    - dmu_numer/denom: (nmix, ncomp)
    - dbeta_numer/denom: (nmix, ncomp)
    - dalpha_numer/denom: (nmix, ncomp)
    - drho_numer/denom: (nmix, ncomp)
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
    
    dAK: NDArray

    loglik_sum: float = 0.0

    newton: Optional[AmicaNewtonUpdates] = None

@dataclass(slots=True, repr=False)
class AmicaNewtonUpdates:
    """Additional accumulators for Newton updates.

    Shapes:
    - dbaralpha_numer/denom: (n_mixtures, n_components, n_models)
    - dkappa_numer/denom: (n_mixtures, n_components, n_models)
    - dlambda_numer/denom: (n_mixtures, n_components, n_models)
    - dsigma2_numer/denom: (n_mixtures, n_components, n_models)
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

    Intentionally small: capture only the core fields commonly inspected during
    training. Extend conservatively as needs arise.
    """

    iter: int                # 1-based iteration index
    loglik: float            # total log-likelihood for the iteration
    ll_inc: float = 0.0      # improvement vs previous iteration
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
    def loglik_array(self) -> np.ndarray:
        return np.array([m.loglik for m in self.metrics], dtype=float)

    def ll_inc_array(self) -> np.ndarray:
        return np.array([m.ll_inc for m in self.metrics], dtype=float)

    def last(self) -> Optional[IterationMetrics]:
        return self.metrics[-1] if self.metrics else None

# TODO: consider making an IterationContext or IterationMetrics class
# to bundle ephemeral constants that shuttle together across functions.
# e.g. Dsum, sldet, comp_list etc.

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
        W = np.array(seeds["W"], dtype=dtype)
        assert W.shape == (num_comps, num_comps, num_models), (
            f"W seed shape {W.shape} != {(num_comps, num_comps, num_models)}"
        )
    else:
        W = np.empty((num_comps, num_comps, num_models))  # Weights for each model

    # A - match amica.py: A = np.zeros((num_comps, num_comps))  
    A = np.zeros((num_comps, num_comps))

    # sbeta, mu, rho - match amica.py patterns
    if "sbeta" in seeds:
        sbeta = np.array(seeds["sbeta"], dtype=dtype)
        assert sbeta.shape == (num_mix, num_comps)
    else:
        sbeta = np.empty((num_mix, num_comps))

    if "mu" in seeds:
        mu = np.array(seeds["mu"], dtype=dtype)  
        assert mu.shape == (num_mix, num_comps)
    else:
        mu = np.zeros((num_mix, num_comps))

    rho = np.full((num_mix, num_comps), rho0, dtype=dtype)  # Shape parameters
    
    # Initialize alpha (mixing coefficients) to zeros - will be computed in first iteration
    alpha = np.zeros((num_mix, num_comps), dtype=dtype)

    gm = np.full(num_models, 1.0 / num_models, dtype=dtype)  # Uniform initialization
    return AmicaState(W=W, A=A, mu=mu, sbeta=sbeta, rho=rho, alpha=alpha, gm=gm)


def get_workspace(cfg: AmicaConfig, *, block_size: Optional[int] = None) -> AmicaWorkspace:
    """Create a workspace with the configured dtype and block size.

    The workspace provides `get(name, shape, dtype=None, init='empty')` to
    request buffers lazily and reuse them across iterations.
    """
    bsize = cfg.block_size if block_size is None else block_size
    return AmicaWorkspace(block_size=bsize, dtype=cfg.dtype)


# TODO: consider making this a class method of AmicaUpdates
def initialize_updates(cfg: AmicaConfig, do_newton: bool=False) -> AmicaUpdates:
    """Allocate zeroed update accumulators with shapes from the config."""
    num_comps = cfg.n_components
    num_models = cfg.n_models  
    num_mix = cfg.n_mixtures
    dtype = cfg.dtype
    shape_2 = (num_mix, num_comps)

    # Match amica.py initialization patterns
    dgm_numer = np.zeros(num_models, dtype=dtype)

    # Update accumulators - match amica.py: (num_mix, num_comps) shape
    dmu_numer = np.zeros(shape_2, dtype=dtype)
    dmu_denom = np.zeros(shape_2, dtype=dtype)

    dbeta_numer = np.zeros(shape_2, dtype=dtype)
    dbeta_denom = np.zeros(shape_2, dtype=dtype)

    dalpha_numer = np.zeros(shape_2, dtype=dtype)
    dalpha_denom = np.zeros(shape_2, dtype=dtype)

    drho_numer = np.zeros(shape_2, dtype=dtype)
    drho_denom = np.zeros(shape_2, dtype=dtype)

    dc_numer = np.zeros((num_comps, num_models), dtype=dtype)
    dc_denom = np.zeros((num_comps, num_models), dtype=dtype)

    dAK = np.zeros((num_comps, num_comps), dtype=np.float64)  # Derivative of A

    if do_newton:
        # NOTE: Amica authors gave newton arrays 3 dims, but gradient descent 2 dims
        shape_3 = (num_mix, num_comps, num_models)

        dbaralpha_numer = np.zeros(shape_3, dtype=dtype)
        dbaralpha_denom = np.zeros(shape_3, dtype=dtype)

        dkappa_numer = np.zeros(shape_3, dtype=dtype)
        dkappa_denom = np.zeros(shape_3, dtype=dtype)

        dlambda_numer = np.zeros(shape_3, dtype=dtype)
        dlambda_denom = np.zeros(shape_3, dtype=dtype)

        # These are 2D in the Fortran code, which actually uses nw x num_models
        dsigma2_numer = np.zeros((num_comps, num_models), dtype=dtype)
        dsigma2_denom = np.zeros((num_comps, num_models), dtype=dtype)

        newton = AmicaNewtonUpdates(
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

    return AmicaUpdates(
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
        dAK=dAK,
        loglik_sum=0.0,
        newton=newton,
    )


def reset_updates(u: AmicaUpdates) -> None:
    """Zero all per-iteration accumulators in-place.

    This avoids reallocations by reusing the same AmicaUpdates instance across
    iterations. It resets numerators/denominators, the weight gradients, the
    mixture numerators, and Newton accumulators if present.
    """
    # Core accumulators

    u.dmu_numer.fill(0.0)
    u.dmu_denom.fill(0.0)

    u.dbeta_numer.fill(0.0)
    u.dbeta_denom.fill(0.0)

    u.dalpha_numer.fill(0.0)
    u.dalpha_denom.fill(0.0)

    u.drho_numer.fill(0.0)
    u.drho_denom.fill(0.0)

    u.dgm_numer.fill(0.0)

    # Scalar diagnostics
    u.loglik_sum = 0.0

    # Optional Newton accumulators
    if u.newton is not None:
        n = u.newton
        n.dbaralpha_numer.fill(0.0)
        n.dbaralpha_denom.fill(0.0)
        n.dkappa_numer.fill(0.0)
        n.dkappa_denom.fill(0.0)
        n.dlambda_numer.fill(0.0)
        n.dlambda_denom.fill(0.0)
        n.dsigma2_numer.fill(0.0)
        n.dsigma2_denom.fill(0.0)


__all__ = [
    "AmicaConfig",
    "AmicaState",
    "AmicaWorkspace",
    "AmicaUpdates",
    "AmicaMetrics",
    "get_initial_state",
    "get_workspace",
    "initialize_updates",
    "reset_updates",
]
