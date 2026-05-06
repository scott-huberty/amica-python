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
        torch.ones(
            state.alpha.shape[0],
            dtype=state.alpha.dtype,
            device=state.alpha.device,
        ),
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
    """Anderson / DAAREM accelerator for EM-like fixed-point maps.

    ``monotone=False`` uses the plain Anderson path. ``monotone=True`` uses
    the DAAREM proposal path validated against the R ``daarem`` package.

    Parameter mapping to R package ``daarem(..., control=list(...))``:

    - ``order`` maps to R ``control$order``. R then uses
      ``min(order, ceiling(num.params / 2))`` as ``nlag``; our AMICA wrapper
      already passes high-dimensional packed states, so we use ``order`` as the
      direct history cap.
    - ``daarem_alpha`` maps to R ``control$alpha``.
    - ``daarem_kappa`` maps to R ``control$kappa``.
    - ``epsilon_monotone`` is the closest Python sibling of R ``mon.tol`` when
      AMICA candidate validation is enabled. R checks monotonicity inside the
      DAAREM loop; AMICA checks it in the shared acceleration wrapper after a
      candidate state is unpacked and scored.
    - ``cycl_monotone_tol`` maps to R ``control$cycl.mon.tol``.

    R controls handled outside this object:

    - ``maxiter`` and ``tol`` are handled by AMICA's outer solver.
    - ``convtype`` is handled by AMICA convergence criteria, not DAAREM.
    - ``intermed`` is handled by AMICA logging/history, not this object.

    R controls not directly implemented:

    - ``resid.tol`` has no direct Python sibling. AMICA relies on common
      candidate validation and reject/restart behavior instead of R's
      objective-free residual-change safeguard.

    Python-only controls with no R sibling:

    - ``damping`` and ``ridge`` apply only to the plain Anderson path
      (``monotone=False``), not the DAAREM path.
    - ``restart_on_reject`` and ``max_consecutive_rejects`` are AMICA wrapper
      safeguards for rejected candidates.
    - ``shrink_count``, ``lambda_ridge``, and ``r_penalty`` are DAAREM's
      adaptive damping state, corresponding to R's ``shrink.count``,
      ``lambda.ridge``, and ``r.penalty`` locals.
    """

    order: int = 5
    damping: float = 1.0
    ridge: float = 1e-8
    monotone: bool = True
    epsilon_monotone: float = 0.0
    restart_on_reject: bool = True
    max_consecutive_rejects: int = 3
    daarem_alpha: float = 1.2
    daarem_kappa: int = 25
    cycl_monotone_tol: float = 0.0
    shrink_count: int = 0
    lambda_ridge: float = 100000.0
    r_penalty: float = 0.0
    cycle_count: int = 0
    cycle_loglik: float | None = None
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
        self.shrink_count = 0
        self.lambda_ridge = 100000.0
        self.r_penalty = 0.0
        self.cycle_count = 0
        self.cycle_loglik = None

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
        if self.monotone:
            return self._propose_daarem(mk)

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
        gram += self.ridge * torch.eye(
            gram.shape[0],
            dtype=gram.dtype,
            device=gram.device,
        )
        rhs = F.T @ fk
        gamma = torch.linalg.solve(gram, rhs)
        raw = gk - (X + F) @ gamma
        candidate = gk + self.damping * (raw - gk)
        return AndersonProposal(candidate=candidate, history=mk)

    def _propose_daarem(self, history: int) -> AndersonProposal | None:
        fnew = self.f_hist[-1]
        delta_f = []
        delta_x = []
        start = len(self.f_hist) - history - 1
        for idx in range(start + 1, len(self.f_hist)):
            delta_f.append(self.f_hist[idx] - self.f_hist[idx - 1])
            delta_x.append(self.x_hist[idx] - self.x_hist[idx - 1])

        F = torch.stack(delta_f, dim=1)
        X = torch.stack(delta_x, dim=1)
        U, singular_values, Vh = torch.linalg.svd(F, full_matrices=False)
        positive = singular_values > 0
        if not torch.any(positive):
            return None
        singular_values = singular_values[positive]
        U = U[:, positive]
        Vh = Vh[positive, :]
        uy = U.T @ fnew
        uy_sq = uy * uy
        ftf = torch.sqrt(torch.sum(uy_sq * singular_values * singular_values))
        self.lambda_ridge, self.r_penalty = _damping_find(
            uy_sq=uy_sq,
            singular_values=singular_values,
            alpha=self.daarem_alpha,
            kappa=self.daarem_kappa,
            shrink_count=self.shrink_count,
            ftf=ftf,
            lambda_start=self.lambda_ridge,
            r_start=self.r_penalty,
        )
        d_sq = singular_values * singular_values
        dd = (singular_values * uy) / (d_sq + self.lambda_ridge)
        gamma = Vh.T @ dd

        xnew = self.x_hist[-1]
        xbar = xnew - X @ gamma
        fbar = fnew - F @ gamma
        return AndersonProposal(candidate=xbar + fbar, history=history)

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
        if self.monotone:
            self.shrink_count += 1

    def update_cycle_monotonicity(self, *, loglik: float, history: int) -> None:
        """Apply R DAAREM's cycle-level monotonicity damping adjustment."""
        if not self.monotone:
            return
        if self.cycle_loglik is None:
            self.cycle_loglik = loglik
        self.cycle_count += 1
        if self.cycle_count != history:
            return

        if loglik < self.cycle_loglik - self.cycl_monotone_tol:
            self.shrink_count = max(
                self.shrink_count - history,
                -2 * self.daarem_kappa,
            )
        self.cycle_loglik = loglik
        self.cycle_count = 0


def _damping_find(
        *,
        uy_sq: torch.Tensor,
        singular_values: torch.Tensor,
        alpha: float,
        kappa: int,
        shrink_count: int,
        ftf: torch.Tensor,
        lambda_start: float | None = None,
        r_start: float | None = None,
        maxit: int = 10,
) -> tuple[float, float]:
    """Port of DAAREM's ``DampingFind`` ridge-penalty search.

    ``alpha``, ``kappa``, and ``shrink_count`` correspond to R's ``aa``,
    ``kappa``, and ``sk`` arguments. ``lambda_start`` and ``r_start``
    correspond to R's warm-started ``lambda.start`` and ``r.start``.
    ``maxit`` maps to R's hard-coded default of 10 damping-search iterations.
    """
    if lambda_start is None or not torch.isfinite(torch.tensor(lambda_start)):
        lambda_start = 100.0
    if r_start is None or not torch.isfinite(torch.tensor(r_start)):
        r_start = 0.0

    dtype = singular_values.dtype
    device = singular_values.device
    dvec = singular_values
    uy_sq = uy_sq.to(dtype=dtype, device=device)
    d_sq = dvec * dvec
    pow_value = kappa - shrink_count
    target = torch.exp(
        -0.5 * torch.log1p(torch.tensor(alpha ** pow_value, dtype=dtype, device=device))
    )
    betahat_ls = uy_sq / d_sq
    betahat_ls_norm = torch.sqrt(torch.sum(betahat_ls))
    vk = target * betahat_ls_norm
    if float(vk) == 0.0:
        return float(lambda_start), float(r_start)

    lambda_value = torch.tensor(lambda_start, dtype=dtype, device=device) - (
        torch.tensor(r_start, dtype=dtype, device=device) / vk
    )
    lower = (
        betahat_ls_norm * (betahat_ls_norm - vk)
    ) / torch.sum(uy_sq / (d_sq * d_sq))
    upper = ftf / vk

    lower_stop = torch.exp(
        -0.5
        * torch.log1p(
            torch.tensor(alpha ** (pow_value + 0.5), dtype=dtype, device=device)
        )
    )
    upper_stop = torch.exp(
        -0.5
        * torch.log1p(
            torch.tensor(alpha ** (pow_value - 0.5), dtype=dtype, device=device)
        )
    )

    s_norm = torch.tensor(0.0, dtype=dtype, device=device)
    phi_ratio = torch.tensor(0.0, dtype=dtype, device=device)
    for _ in range(maxit):
        if lambda_value <= lower or lambda_value >= upper:
            lambda_value = torch.maximum(
                0.0001 * upper,
                torch.sqrt(lower * upper),
            )

        d_lambda = (dvec / (d_sq + lambda_value)) ** 2
        d_prime = d_lambda / (d_sq + lambda_value)
        s_norm = torch.sqrt(torch.sum(uy_sq * d_lambda))
        phi_val = s_norm - vk
        phi_der = -torch.sum(uy_sq * d_prime) / s_norm
        phi_ratio = phi_val / phi_der

        if (
            s_norm <= upper_stop * betahat_ls_norm
            and s_norm >= lower_stop * betahat_ls_norm
        ):
            break

        upper = torch.where(phi_val >= 0, upper, lambda_value)
        lower = torch.maximum(lower, lambda_value - phi_ratio)
        lambda_value = lambda_value - (s_norm * phi_ratio) / vk

    return float(lambda_value), float(s_norm * phi_ratio)
