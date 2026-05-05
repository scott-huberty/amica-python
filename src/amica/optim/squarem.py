"""SQUAREM acceleration for fixed-point EM maps.

The implementation follows the K=1 SQUAREM update in the R package:

``p_sq = p + 2 * alpha * (p1 - p) + alpha**2 * ((p2 - p1) - (p1 - p))``

where ``p1 = G(p)`` and ``p2 = G(p1)``.

Parameter mapping to R package ``SQUAREM::squarem(..., control=list(...))``:

- ``method`` maps to R ``control$method``. We support the K=1 numeric methods
  ``1``, ``2``, and ``3`` and default to R's K=1 default, ``3``.
- ``step_min`` maps to R ``control$step.min0``.
- ``step_max`` maps to R ``control$step.max0`` initially, then follows R's
  adaptive ``step.max`` behavior after accepted/rejected bounded steps.
- ``mstep`` maps to R ``control$mstep``.

R controls intentionally not exposed here:

- ``K`` is fixed at ``1``. R's K > 1 cycled variants are not implemented.
- ``square`` only applies to R's K > 1 cycled variants, so it has no Python
  equivalent here.
- ``tol`` and ``maxiter`` are handled by AMICA's outer solver, not the
  accelerator object.
- ``trace`` and ``intermed`` are handled by AMICA logging/history, not this
  object.
- ``objfn.inc`` and ``minimize`` are represented by AMICA's common candidate
  validation path, especially ``accelerator_validate_candidate``. They are not
  parameters of ``SQUAREMAccelerator``.
- ``kr`` is used by R's objective-free ``squarem2`` residual safeguard. Our
  AMICA integration uses common validity and optional likelihood validation
  instead, so there is no direct sibling.

Python-only state with no R control sibling:

- ``x_hist`` and ``g_hist`` store the two fixed-point evaluations supplied by
  AMICA's shared acceleration wrapper.
- ``last_alpha`` stores the most recent clipped step length so accept/reject can
  update ``step_max`` like R.
"""

from __future__ import annotations  # until min>=3.14 https://peps.python.org/pep-0649/

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SQUAREMProposal:
    """One SQUAREM candidate and associated metadata."""

    candidate: torch.Tensor
    history: int
    alpha: float


@dataclass(slots=True)
class SQUAREMAccelerator:
    """K=1 SQUAREM accelerator matching the R package step formulas.

    The R implementation supports K={2,3} higher-order schemes, which "may provide
    greater speed in some problems, although they are less reliable than first-order
    (K=1) schemes". We only implement K=1 first-order schemes. FYI this corresponds to
    the ``controls$K`` parameter in R's ``squarem``.
    
    Parameters
    ----------
    method : int
        An integer that denotes the particular SQUAREM scheme to be used.
        Must be one of ``{1, 2, 3}`` (defaults to ``3``.)
        These correspond to the 3 schemes discussed in Varadhan and Roland (2008)
        (see below). These three schemes differ only in how the scalar step length is
        computed:
        
        * 1
          SqS1: ``alpha = (r.T @ v) / (v.T @ v)          # equation 7``
        * 2
          SqS2: ``alpha = (r.T @ r) / (r.T @ v)          # equation 8``
        * 3
          SqS3: ``alpha = -||r|| / ||v||                 # equation 9``

        This corresponds to ``control$method`` in the R ``squarem`` API.
    step_min: int
        maps to R ``control$step.min0``. A scalar denoting the minimum steplength taken
        by a SQUAREM algorithm. Default is 1. For contractive fixed-point iterations
        (e.g., EM and MM), this default works well. In problems where an eigenvalue of
        the Jacobian of F is outside of the interval ``(0,1)``, ``step_min`` should be
        less than ``1`` or even negative in some cases.
    step_max: int
        maps to R ``control$step.max0``. A positive-valued scalar denoting the initial
        value of the maximum steplength taken by a SQUAREM algorithm. Default is ``1``.
        When the steplength computed by SQUAREM exceeds step.max0, the steplength is set
        equal to ``step_max``, but then step.max0 is increased by a factor of ``mstep``.
    mstep: int
        A scalar greater than 1. When the steplength computed by SQUAREM exceeds
        ``step_max``, the steplength is set equal to ``step_max``, but ``step_max`` is
        increased by a factor of ``mstep``. Default is ``4``.
    x_hist : list
        store the two fixed-point evaluations
    y_hist : list
        stores the two fixed-point evaluations.
    last_alpha : list
        stores the most recent clipped step length so accept/reject can update
        ``step_max`` like R.
    """

    method: int = 3
    step_min: float = 1.0
    step_max: float = 1.0
    mstep: float = 4.0
    x_hist: list[torch.Tensor] | None = None
    g_hist: list[torch.Tensor] | None = None
    last_alpha: float | None = None

    def __post_init__(self) -> None:
        """Initialize mutable history buffers."""
        if self.method not in {1, 2, 3}:
            raise ValueError("SQUAREM method must be 1, 2, or 3.")
        if self.x_hist is None:
            self.x_hist = []
        if self.g_hist is None:
            self.g_hist = []

    def reset(self) -> None:
        """Clear stored fixed-point history."""
        self.x_hist.clear()
        self.g_hist.clear()
        self.last_alpha = None

    def update(self, *, x: torch.Tensor, g: torch.Tensor) -> None:
        """Record one fixed-point evaluation ``g = G(x)``."""
        self.x_hist.append(x.detach().clone().to(dtype=torch.float64))
        self.g_hist.append(g.detach().clone().to(dtype=torch.float64))
        if len(self.x_hist) > 2:
            self.x_hist = self.x_hist[-2:]
            self.g_hist = self.g_hist[-2:]

    def propose(self) -> SQUAREMProposal | None:
        """Return a SQUAREM candidate once two consecutive EM steps exist."""
        if len(self.x_hist) < 2:
            return None

        p = self.x_hist[-2]
        p1 = self.g_hist[-2]
        p1_as_next_x = self.x_hist[-1]
        p2 = self.g_hist[-1]
        if not torch.allclose(p1, p1_as_next_x):
            return None

        q1 = p1 - p
        q2 = p2 - p1
        v = q2 - q1
        sr2 = torch.dot(q1, q1)
        sv2 = torch.dot(v, v)
        srv = torch.dot(q1, v)

        alpha = self._compute_alpha(sr2=sr2, sv2=sv2, srv=srv)
        if alpha is None:
            return None
        alpha = max(self.step_min, min(self.step_max, alpha))
        self.last_alpha = alpha

        candidate = p + 2.0 * alpha * q1 + (alpha * alpha) * v
        return SQUAREMProposal(candidate=candidate, history=2, alpha=alpha)

    def _compute_alpha(
            self,
            *,
            sr2: torch.Tensor,
            sv2: torch.Tensor,
            srv: torch.Tensor,
    ) -> float | None:
        tiny = torch.finfo(sr2.dtype).tiny
        if self.method == 1:
            if torch.abs(sv2) <= tiny:
                return None
            return float(-srv / sv2)
        if self.method == 2:
            if torch.abs(srv) <= tiny:
                return None
            return float(-sr2 / srv)
        if sv2 <= tiny:
            return None
        return float(torch.sqrt(sr2 / sv2))

    def reject(self) -> bool:
        """Reject the candidate and restart the short SQUAREM history."""
        alpha = self.last_alpha
        self.reset()
        if alpha == self.step_max and self.step_max > self.step_min:
            self.step_max = max(self.step_min, self.step_max / self.mstep)
        return True

    def accept(self) -> None:
        """Accept the candidate and keep only the latest EM evaluation."""
        if self.last_alpha == self.step_max and self.step_max > 0:
            self.step_max *= self.mstep
        self.x_hist = self.x_hist[-1:]
        self.g_hist = self.g_hist[-1:]
