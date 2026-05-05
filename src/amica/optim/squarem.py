"""SQUAREM acceleration for fixed-point EM maps.

The implementation follows the K=1 SQUAREM update in the R package:

``p_sq = p + 2 * alpha * (p1 - p) + alpha**2 * ((p2 - p1) - (p1 - p))``

where ``p1 = G(p)`` and ``p2 = G(p1)``.
"""

from __future__ import annotations

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
    """K=1 SQUAREM accelerator matching the R package step formulas."""

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
