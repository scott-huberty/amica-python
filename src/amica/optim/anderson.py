"""Anderson / DAAREM-style acceleration around a fixed-point EM map.

This accelerator treats one AMICA EM iteration as a fixed-point map x <- G(x)
and applies damped Anderson-style extrapolation to that map.

References
----------
Henderson & Varadhan, "Damped Anderson acceleration with restarts and
epsilon-monotonicity for accelerating EM and EM-like algorithms"
Varadhan & Roland, "Simple and globally convergent methods for accelerating
the convergence of any EM algorithm"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from amica.linalg import get_unmixing_matrices
from amica.state import AmicaState


def _softmax_last_dim(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _logits_from_simplex(probs: torch.Tensor) -> torch.Tensor:
    last = probs[..., -1:].clamp_min(torch.finfo(probs.dtype).tiny)
    head = probs[..., :-1].clamp_min(torch.finfo(probs.dtype).tiny)
    return torch.log(head) - torch.log(last)


def pack_state(state: AmicaState) -> torch.Tensor:
    """Pack free AMICA parameters into one unconstrained vector."""
    parts = [
        state.A.reshape(-1),
        state.c.reshape(-1),
        state.mu.reshape(-1),
        torch.log(state.sbeta).reshape(-1),
        torch.log(state.rho).reshape(-1),
        _logits_from_simplex(state.alpha).reshape(-1),
    ]
    return torch.cat(parts).to(dtype=torch.float64)


def unpack_state(
        vec: torch.Tensor,
        template_state: AmicaState,
) -> AmicaState:
    """Unpack a vector back into an AMICA parameter state."""
    state = template_state.clone()
    dtype = template_state.A.dtype
    device = template_state.A.device
    vec = vec.to(device=device, dtype=dtype)

    n_comp, n_mix = state.alpha.shape
    a_size = state.A.numel()
    c_size = state.c.numel()
    mu_size = state.mu.numel()
    sbeta_size = state.sbeta.numel()
    rho_size = state.rho.numel()
    alpha_logits_size = n_comp * (n_mix - 1)
    expected = a_size + c_size + mu_size + sbeta_size + rho_size + alpha_logits_size
    if vec.numel() != expected:
        raise ValueError(f"Packed state has {vec.numel()} values, expected {expected}.")

    offset = 0
    state.A = vec[offset:offset + a_size].reshape_as(state.A).clone()
    offset += a_size
    state.c = vec[offset:offset + c_size].reshape_as(state.c).clone()
    offset += c_size
    state.mu = vec[offset:offset + mu_size].reshape_as(state.mu).clone()
    offset += mu_size
    state.sbeta = torch.exp(
        vec[offset:offset + sbeta_size].reshape_as(state.sbeta)
    ).clone()
    offset += sbeta_size
    state.rho = torch.exp(
        vec[offset:offset + rho_size].reshape_as(state.rho)
    ).clone()
    offset += rho_size
    alpha_logits = vec[offset:offset + alpha_logits_size].reshape(n_comp, n_mix - 1)
    alpha_logits = torch.cat(
        [alpha_logits, torch.zeros((n_comp, 1), dtype=dtype, device=device)],
        dim=1,
    )
    state.alpha = _softmax_last_dim(alpha_logits).clone()

    state.W, _ = get_unmixing_matrices(c=state.c, A=state.A, W=state.W)
    state.gm = template_state.gm.clone()
    return state


def is_valid_state(state: AmicaState, *, atol: float = 1e-8) -> tuple[bool, str]:
    """Run cheap sanity checks before accepting an accelerated candidate."""
    fields = state.to_dict()
    for name, value in fields.items():
        if not torch.all(torch.isfinite(value)):
            return False, f"nonfinite_{name}"
    if torch.any(state.sbeta <= 0):
        return False, "nonpositive_sbeta"
    if torch.any(state.rho <= 0):
        return False, "nonpositive_rho"
    if torch.any(state.alpha <= 0):
        return False, "nonpositive_alpha"
    if not torch.allclose(
        state.alpha.sum(dim=1),
        torch.ones(state.alpha.shape[0], dtype=state.alpha.dtype, device=state.alpha.device),
        atol=atol,
        rtol=0.0,
    ):
        return False, "alpha_not_simplex"
    try:
        W, _ = get_unmixing_matrices(c=state.c, A=state.A, W=state.W)
    except Exception:
        return False, "invalid_unmixing"
    if not torch.all(torch.isfinite(W)):
        return False, "nonfinite_W"
    return True, "ok"


@dataclass(slots=True)
class AndersonProposal:
    """One accelerated candidate and associated metadata."""

    candidate: torch.Tensor
    history: int


@dataclass(slots=True)
class AndersonEMAccelerator:
    """Minimal damped Anderson accelerator for EM-like fixed-point maps."""

    order: int = 5
    damping: float = 1.0
    ridge: float = 1e-8
    monotone: bool = True
    epsilon_monotone: float = 0.0
    restart_on_reject: bool = True
    max_consecutive_rejects: int = 3
    x_hist: list[torch.Tensor] = field(default_factory=list)
    g_hist: list[torch.Tensor] = field(default_factory=list)
    f_hist: list[torch.Tensor] = field(default_factory=list)
    consecutive_rejects: int = 0
    restart_count: int = 0

    def reset(self) -> None:
        self.x_hist.clear()
        self.g_hist.clear()
        self.f_hist.clear()
        self.consecutive_rejects = 0
        self.restart_count = 0

    def update(self, *, x: torch.Tensor, g: torch.Tensor) -> None:
        f = g - x
        self.x_hist.append(x.detach().clone().to(dtype=torch.float64))
        self.g_hist.append(g.detach().clone().to(dtype=torch.float64))
        self.f_hist.append(f.detach().clone().to(dtype=torch.float64))
        keep = self.order + 1
        if len(self.x_hist) > keep:
            self.x_hist = self.x_hist[-keep:]
            self.g_hist = self.g_hist[-keep:]
            self.f_hist = self.f_hist[-keep:]

    def propose(self) -> AndersonProposal | None:
        if len(self.x_hist) < 2:
            return None
        mk = min(self.order, len(self.x_hist) - 1)
        if mk <= 0:
            return None

        xk = self.x_hist[-1]
        gk = self.g_hist[-1]
        fk = self.f_hist[-1]

        delta_f = []
        delta_x = []
        start = len(self.f_hist) - mk - 1
        for idx in range(start + 1, len(self.f_hist)):
            delta_f.append(self.f_hist[idx] - self.f_hist[idx - 1])
            delta_x.append(self.x_hist[idx] - self.x_hist[idx - 1])

        F = torch.stack(delta_f, dim=1)
        X = torch.stack(delta_x, dim=1)
        gram = F.T @ F
        gram += self.ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        rhs = F.T @ fk
        gamma = torch.linalg.solve(gram, rhs)
        raw = gk - (X + F) @ gamma
        candidate = gk + self.damping * (raw - gk)
        return AndersonProposal(candidate=candidate, history=mk)

    def reject(self) -> bool:
        self.consecutive_rejects += 1
        should_restart = self.restart_on_reject or (
            self.consecutive_rejects >= self.max_consecutive_rejects
        )
        if should_restart:
            self.x_hist = self.x_hist[-1:]
            self.g_hist = self.g_hist[-1:]
            self.f_hist = self.f_hist[-1:]
            self.consecutive_rejects = 0
            self.restart_count += 1
        return should_restart

    def accept(self) -> None:
        self.consecutive_rejects = 0
