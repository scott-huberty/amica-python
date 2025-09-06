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

- Shapes: mu/sbeta/rho use (ncomp, nmix), matching new standard (e.g., mu[comp_indices, :]).
- Workspace is intentionally generic (name → buffer). This lets us wire temps one-by-one without locking the shapes prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray

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
    - W:  (ncomp, n_feature, nmix)   unmixing matrices per mixture
    - A:  (n_feature, ncomp, nmix)   mixing matrices per mixture (often inv(W))
    - c: (ncomp, n_models)       bias (offset) terms per component and model.
    - mu: (ncomp, nmix)          location parameters per component and mixture
    - sbeta: (ncomp, nmix)       scale parameters per component and mixture
    - rho: (ncomp, nmix)         shape parameters per component and mixture
    - alpha: (ncomp, nmix)       mixing coefficients per component and mixture
    - gm: (nmix,)                mixture weights (prior over models)
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
        }


@dataclass
class AmicaWorkspace:
    """Enhanced workspace with improved buffer management.

    Provides allocation, validation, and access to reusable temporary buffers.
    Buffers are allocated on-demand with shape/dtype validation to prevent
    silent bugs from mismatched buffer usage.
    """

    dtype: np.dtype = np.float64
    _buffers: Dict[str, NDArray] = field(default_factory=dict)
    
    def allocate_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        dtype: Optional[np.dtype] = None,
        init: str = "zeros",
        force_realloc: bool = False,
    ) -> NDArray:
        """Allocate or reallocate a specific buffer with validation.

        Parameters
        ----------
        name : str
            Unique buffer identifier (e.g., 'b', 'y', 'z0').
        shape : Tuple[int, ...]
            Required buffer shape. Must be positive integers.
        dtype : np.dtype, optional
            Buffer data type. If None, uses workspace default.
        init : str, default='zeros'
            Initialization mode: 'zeros', 'ones', or 'empty'.
        force_realloc : bool, default=False
            If True, forces reallocation even if buffer exists with correct shape/dtype.

        Returns
        -------
        NDArray
            The allocated buffer.

        Raises
        ------
        ValueError
            If shape contains non-positive values or init mode is invalid.
        RuntimeError
            If attempting to change shape/dtype of existing buffer without force_realloc.
        """
        # Input validation
        if not all(isinstance(s, int) and s > 0 for s in shape):
            raise ValueError(f"Buffer '{name}': shape must contain positive integers, got {shape}")
        
        if init not in {"zeros", "ones", "empty"}:
            raise ValueError(f"Buffer '{name}': invalid init mode '{init}', must be 'zeros', 'ones', or 'empty'")

        dtype = self.dtype if dtype is None else dtype
        existing_buffer = self._buffers.get(name)
        
        # Check for shape/dtype conflicts with existing buffer
        if existing_buffer is not None:
            shape_matches = existing_buffer.shape == shape
            dtype_matches = existing_buffer.dtype == dtype
            
            if shape_matches and dtype_matches and not force_realloc:
                # Buffer exists and matches - return it (optionally reinitialize)
                if init == "zeros":
                    existing_buffer.fill(0.0)
                elif init == "ones":
                    existing_buffer.fill(1.0)
                # 'empty' means don't reinitialize
                return existing_buffer
            
            elif not force_realloc:
                # Shape/dtype mismatch without force_realloc
                raise RuntimeError(
                    f"Buffer '{name}' shape/dtype mismatch: "
                    f"existing=({existing_buffer.shape}, {existing_buffer.dtype}), "
                    f"requested=({shape}, {dtype}). Use force_realloc=True to override."
                )

        # Allocate new buffer
        if init == "zeros":
            buffer = np.zeros(shape, dtype=dtype)
        elif init == "ones":
            buffer = np.ones(shape, dtype=dtype)
        else:  # init == "empty"
            buffer = np.empty(shape, dtype=dtype)

        # Store buffer
        self._buffers[name] = buffer
        return buffer

    def get_buffer(self, name: str) -> NDArray:
        """Get an already-allocated buffer.

        Parameters
        ----------
        name : str
            Buffer name.

        Returns
        -------
        NDArray
            The requested buffer.

        Raises
        ------
        KeyError
            If buffer has not been allocated.
        """
        try:
            return self._buffers[name]
        except KeyError:
            raise KeyError(
                f"Buffer '{name}' not allocated. "
                f"Available buffers: {list(self._buffers.keys())}"
            )

    def allocate_all(self, buffer_specs: Dict[str, Tuple[int, ...]], *, init: str = "zeros") -> None:
        """Pre-allocate multiple buffers at once.

        Parameters
        ----------
        buffer_specs : Dict[str, Tuple[int, ...]]
            Mapping of buffer names to their required shapes.
        init : str, default='zeros'
            Initialization mode for all buffers.

        Example
        -------
        >>> workspace.allocate_all({
        ...     "b": (512, 32, 1),
        ...     "y": (512, 32, 3, 1),
        ...     "z": (512, 32, 3, 1),
        ... })
        """
        for name, shape in buffer_specs.items():
            self.allocate_buffer(name, shape, init=init)

    def has_buffer(self, name: str) -> bool:
        """Check if a buffer exists."""
        return name in self._buffers

    def buffer_info(self, name: str) -> Dict[str, any]:
        """Get information about a buffer."""
        if not self.has_buffer(name):
            raise KeyError(f"Buffer '{name}' not allocated.")
        
        buffer = self._buffers[name]
        return {
            "shape": buffer.shape,
            "dtype": buffer.dtype,
            "size": buffer.size,
            "nbytes": buffer.nbytes,
        }

    def clear(self, names: Optional[Iterable[str]] = None) -> None:
        """Drop one or more buffers."""
        if names is None:
            self._buffers.clear()
        else:
            for name in names:
                self._buffers.pop(name, None)
    
    def zero_all(self) -> None:
        """Zero all existing buffers in-place."""
        for arr in self._buffers.values():
            arr.fill(0.0)

    def total_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        return sum(buf.nbytes for buf in self._buffers.values())

    # Legacy compatibility method
    def get(self, name: str, shape: Tuple[int, ...], *, dtype: Optional[np.dtype] = None, init: str = "zeros") -> NDArray:
        """Legacy compatibility method. Use allocate_buffer() for new code."""
        return self.allocate_buffer(name, shape, dtype=dtype, init=init)


@dataclass(slots=True, repr=False)
class AmicaUpdates:
    """Aggregated updates computed in one iteration.

    This container mainly helps to shuttle a number of arrays together
    across the get_updates_and_likelihood and update_params functions.

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
    
    dAK: NDArray
    loglik_sum: float = 0.0

    newton: Optional[AmicaNewtonUpdates] = None

    def reset(self) -> None:
        """Zero all per-iteration accumulators in-place.

        Keeps array allocations stable while clearing their contents for the
        next iteration. Extensible: if new fields are added, include them here.
        """
        # Core accumulators
        self.dmu_numer.fill(0.0)
        self.dmu_denom.fill(0.0)

        self.dbeta_numer.fill(0.0)
        self.dbeta_denom.fill(0.0)

        self.dalpha_numer.fill(0.0)
        self.dalpha_denom.fill(0.0)

        self.drho_numer.fill(0.0)
        self.drho_denom.fill(0.0)

        self.dgm_numer.fill(0.0)

        self.dc_numer.fill(0.0)
        self.dc_denom.fill(0.0)

        self.dAK.fill(0.0)
        self.loglik_sum = 0.0

        # If Newton accumulators are present, zero them as well
        if self.newton is not None:
            self.newton.dbaralpha_numer.fill(0.0)
            self.newton.dbaralpha_denom.fill(0.0)
            self.newton.dkappa_numer.fill(0.0)
            self.newton.dkappa_denom.fill(0.0)
            self.newton.dlambda_numer.fill(0.0)
            self.newton.dlambda_denom.fill(0.0)
            self.newton.dsigma2_numer.fill(0.0)
            self.newton.dsigma2_denom.fill(0.0)

@dataclass(slots=True, repr=False)
class AmicaNewtonUpdates:
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
    - no_newt: If True, Newton updates were disabled for this iteration due to numerical issues. Only relevant if AmicaConfig.do_newton is True.
    """

    iter: int                           # 1-based iteration index
    loglik: Optional[float] = None      # total log-likelihood for the iteration
    ndtmpsum: Optional[float] = None    # normalized update norm for the iteration
    lrate: Optional[float] = None       # learning rate used for the iteration
    rholrate: Optional[float] = None    # rho learning rate used for the iteration
    ll_inc: float = 0.0                 # improvement vs previous iteration
    step_time_s: Optional[float] = None
    numincs: Optional[int] = None
    numdecs: Optional[int] = None
    no_newt: bool = False               # Disable Newton due to numerical issues


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
        W = np.array(seeds["W"], dtype=dtype)
        assert W.shape == (num_comps, num_comps, num_models), (
            f"W seed shape {W.shape} != {(num_comps, num_comps, num_models)}"
        )
    else:
        W = np.empty((num_comps, num_comps, num_models))  # Weights for each model

    # A - match amica.py: A = np.zeros((num_comps, num_comps))  
    A = np.zeros((num_comps, num_comps))

    c = np.zeros((num_comps, num_models))  # Bias terms per component and model
    # sbeta, mu, rho - match amica.py patterns
    if "sbeta" in seeds:
        sbeta = np.array(seeds["sbeta"], dtype=dtype)
        assert sbeta.shape == (num_comps, num_mix)
    else:
        sbeta = np.empty((num_comps, num_mix))

    if "mu" in seeds:
        mu = np.array(seeds["mu"], dtype=dtype)  
        assert mu.shape == (num_comps, num_mix)
    else:
        mu = np.zeros((num_comps, num_mix))

    rho = np.full((num_comps, num_mix), rho0, dtype=dtype)  # Shape parameters
    
    # Initialize alpha (mixing coefficients) to zeros - will be computed in first iteration
    alpha = np.zeros((num_comps, num_mix), dtype=dtype)

    gm = np.full(num_models, 1.0 / num_models, dtype=dtype)  # Uniform initialization
    return AmicaState(W=W, A=A, c=c, mu=mu, sbeta=sbeta, rho=rho, alpha=alpha, gm=gm)


def get_workspace(cfg: AmicaConfig, *, block_size: Optional[int] = None) -> AmicaWorkspace:
    """Create a workspace with the configured dtype and block size.

    The workspace provides `get(name, shape, dtype=None, init='empty')` to
    request buffers lazily and reuse them across iterations.
    """
    bsize = cfg.block_size if block_size is None else block_size
    return AmicaWorkspace(block_size=bsize, dtype=cfg.dtype)


# TODO: consider making this a class method of AmicaUpdates
def initialize_updates(cfg: AmicaConfig) -> AmicaUpdates:
    """Allocate zeroed update accumulators with shapes from the config."""
    num_comps = cfg.n_components
    num_models = cfg.n_models  
    num_mix = cfg.n_mixtures
    dtype = cfg.dtype
    do_newton = cfg.do_newton
    shape_2 = (num_comps, num_mix)

    # Match amica.py initialization patterns
    dgm_numer = np.zeros(num_models, dtype=dtype)

    # Update accumulators - standardized: (num_comps, num_mix) shape
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

    dAK = np.zeros((num_comps, num_comps), dtype=dtype)  # Derivative of A

    if do_newton:
        # NOTE: Amica authors gave newton arrays 3 dims, but gradient descent 2 dims
        shape_3 = (num_comps, num_mix, num_models)

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
