import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from amica.core import (
    DEFAULT_OPTIMIZER_KWARGS,
    _resolve_optimizer_kwargs,
    em_step,
    initialize_state_parameters,
    maybe_apply_acceleration,
    solve,
)
from amica.optim import (
    AndersonEMAccelerator,
    pack_state,
    unpack_state,
)
from amica.state import AmicaConfig, get_initial_state, initialize_accumulators


def _make_config(max_iter: int = 1, optimizer: str = "em") -> AmicaConfig:
    return AmicaConfig(
        n_features=2,
        n_components=2,
        n_models=1,
        n_mixtures=2,
        batch_size=4,
        max_iter=max_iter,
        do_newton=False,
        optimizer=optimizer,
        accelerator_start_iter=1,
        accelerator_period=1,
        verbose=0,
    )


def _make_initial_params():
    initial_weights = np.array([[0.2, 0.8], [0.4, 0.1]], dtype=np.float64)
    initial_scales = np.array([[0.4, 0.6], [0.7, 0.3]], dtype=np.float64)
    initial_locations = np.array([[0.25, 0.75], [0.6, 0.1]], dtype=np.float64)
    return initial_weights, initial_scales, initial_locations


def test_pack_unpack_round_trip():
    cfg = _make_config()
    state = get_initial_state(cfg)
    rng = torch.Generator().manual_seed(0)
    state, _ = initialize_state_parameters(state=state, config=cfg, rng=rng)

    packed = pack_state(state)
    restored = unpack_state(packed, state)

    assert torch.allclose(restored.A, state.A)
    assert torch.allclose(restored.c, state.c)
    assert torch.allclose(restored.mu, state.mu)
    assert torch.allclose(restored.sbeta, state.sbeta)
    assert torch.allclose(restored.rho, state.rho)
    assert torch.allclose(restored.alpha, state.alpha)
    assert torch.allclose(restored.W, state.W)


def test_em_step_matches_single_iteration_solver():
    cfg = _make_config(max_iter=1)
    X = np.array(
        [[0.1, -0.3], [0.5, 0.2], [-0.4, 0.7], [0.8, -0.1]],
        dtype=np.float64,
    )
    initial_weights, initial_scales, initial_locations = _make_initial_params()

    state = get_initial_state(cfg)
    rng = torch.Generator().manual_seed(123)
    state, wc = initialize_state_parameters(
        state=state,
        config=cfg,
        rng=rng,
        initial_weights=initial_weights,
        initial_scales=initial_scales,
        initial_locations=initial_locations,
    )
    step = em_step(
        X=torch.as_tensor(X, dtype=cfg.dtype, device=cfg.device),
        sldet=0.0,
        wc=wc,
        config=cfg,
        state=state,
        iteration=1,
        do_newton=cfg.do_newton,
        accumulators=initialize_accumulators(cfg),
        lrate=cfg.lrate,
        rholrate=cfg.rholrate,
        lrate0=cfg.lrate,
        rholrate0=cfg.rholrate,
        newtrate=cfg.newtrate,
    )

    solved, _ = solve(
        X,
        config=cfg,
        state=get_initial_state(cfg),
        sldet=0.0,
        initial_weights=initial_weights,
        initial_scales=initial_scales,
        initial_locations=initial_locations,
    )

    assert np.allclose(step.state.W.cpu().numpy(), solved["W"])
    assert np.allclose(step.state.A.cpu().numpy(), solved["A"])
    assert np.allclose(step.state.mu.cpu().numpy(), solved["mu"])
    assert np.allclose(step.state.sbeta.cpu().numpy(), solved["sbeta"])
    assert np.allclose(step.state.rho.cpu().numpy(), solved["rho"])
    assert np.allclose(step.state.alpha.cpu().numpy(), solved["alpha"])
    assert np.allclose(step.state.c.cpu().numpy(), solved["c"])


def test_anderson_proposal_improves_toy_fixed_point():
    accelerator = AndersonEMAccelerator(order=3, damping=1.0, ridge=1e-8)
    x = torch.tensor([8.0], dtype=torch.float64)
    for _ in range(4):
        g = 0.5 * x + 1.0
        accelerator.update(x=x, g=g)
        x = g

    proposal = accelerator.propose()
    assert proposal is not None
    plain = g.item()
    accelerated = proposal.candidate.item()
    assert np.isfinite(accelerated)
    assert abs(accelerated - 2.0) <= abs(plain - 2.0)


def test_daarem_proposal_matches_local_r_daarem_package():
    os.environ.setdefault("R_HOME", "/Users/scotterik/miniforge3/envs/amica_env/lib/R")
    os.environ["PATH"] = (
        "/Users/scotterik/miniforge3/envs/amica_env/bin:" + os.environ.get("PATH", "")
    )
    try:
        from rpy2 import robjects
    except Exception as exc:
        pytest.skip(f"rpy2/R unavailable: {exc}")

    r_dir = "/Users/scotterik/devel/projects/amica-python/optimizers/daarem/R"
    for name in (
        "DampingFind.R",
        "daarem_base_noobjfn.R",
        "daarem_base_objfn.R",
        "daarem.R",
    ):
        robjects.r(f'source("{r_dir}/{name}")')
    robjects.r(
        "fixpt <- function(par) c("
        "0.5 * par[1] - 0.5,"
        "0.5 * par[2] + 0.5,"
        "0.5 * par[3] + 0.5)"
    )
    robjects.r(
        "out <- daarem("
        "par=c(1, -0.5, 2),"
        "fixptfn=fixpt,"
        "control=list(maxiter=3, order=3, tol=1e-12)"
        ")"
    )
    r_par = np.asarray(robjects.r("out$par"), dtype=np.float64)

    accelerator = AndersonEMAccelerator(order=3, monotone=True)
    p = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    p1 = torch.tensor([0.0, 0.25, 1.5], dtype=torch.float64)
    p2 = torch.tensor([-0.5, 0.625, 1.25], dtype=torch.float64)
    accelerator.update(x=p, g=p1)
    accelerator.update(x=p1, g=p2)
    proposal = accelerator.propose()

    assert proposal is not None
    assert np.allclose(proposal.candidate.numpy(), r_par)


def test_daarem_accelerator_defaults_match_r_reference_controls():
    accelerator = AndersonEMAccelerator(monotone=True)

    assert accelerator.order == 1
    assert accelerator.epsilon_monotone == pytest.approx(0.01)
    assert accelerator.daarem_alpha == pytest.approx(1.2)
    assert accelerator.daarem_kappa == 25
    assert accelerator.cycl_monotone_tol == pytest.approx(0.0)

@pytest.mark.xfail(reason="Review whether this is needed.")
def test_optimizer_kwargs_defaults_use_single_step_acceleration():
    kwargs = _resolve_optimizer_kwargs(None)

    assert kwargs["accelerator_order"] == 1
    assert kwargs["accelerator_period"] == 1
    assert DEFAULT_OPTIMIZER_KWARGS["accelerator_order"] == 1
    assert DEFAULT_OPTIMIZER_KWARGS["accelerator_period"] == 1


def test_optimizer_kwargs_reject_unknown_keys():
    with pytest.raises(TypeError, match="Unknown optimizer_kwargs keys"):
        _resolve_optimizer_kwargs({"accelerator_orders": 2})


def test_daarem_cycle_monotonicity_increases_damping_after_cycle_drop():
    accelerator = AndersonEMAccelerator(
        order=3,
        monotone=True,
        daarem_kappa=25,
        cycl_monotone_tol=0.0,
    )

    accelerator.update_cycle_monotonicity(loglik=10.0, history=3)
    accelerator.accept()
    accelerator.update_cycle_monotonicity(loglik=11.0, history=3)
    accelerator.accept()
    accelerator.update_cycle_monotonicity(loglik=9.0, history=3)

    assert accelerator.shrink_count == -1
    assert accelerator.cycle_count == 0
    assert accelerator.cycle_loglik == 9.0


def test_daarem_resets_history_after_full_r_cycle():
    accelerator = AndersonEMAccelerator(order=3, monotone=True)

    for idx in range(4):
        x = torch.tensor([float(idx), float(idx * idx)], dtype=torch.float64)
        g = x + torch.tensor([1.0 + idx, 0.5], dtype=torch.float64)
        accelerator.update(x=x, g=g)
        proposal = accelerator.propose()
        if proposal is None:
            continue
        accelerator.accept()
        accelerator.update_cycle_monotonicity(
            loglik=float(idx),
            history=proposal.history,
        )

    assert accelerator.cycle_count == 0
    assert len(accelerator.x_hist) == 1
    assert torch.allclose(
        accelerator.x_hist[0],
        torch.tensor([3.0, 9.0], dtype=torch.float64),
    )


def test_rejected_candidate_falls_back_to_plain_em(monkeypatch):
    cfg = _make_config(optimizer="anderson")
    accelerator = AndersonEMAccelerator(order=2, damping=1.0, ridge=1e-8)

    previous_state = get_initial_state(cfg)
    current_state = get_initial_state(cfg)
    rng = torch.Generator().manual_seed(0)
    previous_state, _ = initialize_state_parameters(
        state=previous_state,
        config=cfg,
        rng=rng,
    )
    current_state = previous_state.clone()
    current_state.sbeta.fill_(2.0)

    bad_state = current_state.clone()
    bad_state.sbeta.fill_(-1.0)

    monkeypatch.setattr(
        "amica.core.unpack_state",
        lambda vec, template_state: bad_state,
    )

    accelerator.update(x=pack_state(previous_state), g=pack_state(current_state))
    accelerator.update(
        x=pack_state(current_state),
        g=pack_state(current_state) + 0.25,
    )

    out_state, _, outcome = maybe_apply_acceleration(
        accelerator=accelerator,
        config=cfg,
        X=torch.zeros((4, 2), dtype=cfg.dtype, device=cfg.device),
        sldet=0.0,
        previous_state=previous_state,
        current_state=current_state,
        current_loglik=0.0,
        iteration=1,
    )

    assert out_state is current_state
    assert not outcome.accepted
    assert outcome.reason == "nonpositive_sbeta"
    assert outcome.restart
    assert len(accelerator.x_hist) == 1


def test_daarem_validated_candidate_updates_cycle_monotonicity(monkeypatch):
    cfg = _make_config(optimizer="daarem")
    cfg = replace(cfg, accelerator_validate_candidate=True)
    accelerator = AndersonEMAccelerator(order=1, monotone=True)

    previous_state = get_initial_state(cfg)
    current_state = get_initial_state(cfg)
    rng = torch.Generator().manual_seed(0)
    previous_state, _ = initialize_state_parameters(
        state=previous_state,
        config=cfg,
        rng=rng,
    )
    current_state = previous_state.clone()
    current_state.sbeta.fill_(2.0)

    previous_vec = pack_state(previous_state)
    accelerator.update(x=previous_vec - 0.1, g=previous_vec)

    monkeypatch.setattr("amica.core.evaluate_loglikelihood", lambda **kwargs: 1.0)

    _, _, outcome = maybe_apply_acceleration(
        accelerator=accelerator,
        config=cfg,
        X=torch.zeros((4, 2), dtype=cfg.dtype, device=cfg.device),
        sldet=0.0,
        previous_state=previous_state,
        current_state=current_state,
        current_loglik=0.0,
        iteration=1,
    )

    assert outcome.accepted
    assert accelerator.cycle_count == 0
    assert accelerator.cycle_loglik == 1.0
