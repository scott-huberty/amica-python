import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from amica.optim import AndersonEMAccelerator
from amica.utils import fetch_daarem_mixdata

rscript = shutil.which("Rscript")
if rscript is None:
    pytest.skip(
        "Rscript not found in PATH; skipping tests that require external R",
        allow_module_level=True,
    )


def _clear_invalid_r_home() -> None:
    r_home = os.environ.get("R_HOME")
    if r_home is not None and not (Path(r_home) / "bin" / "Rscript").exists():
        os.environ.pop("R_HOME")


def _make_likelihood_matrix(n_samples=1000, n_mixtures=30, random_state=0):
    rng = np.random.default_rng(random_state)
    weights = rng.dirichlet(np.full(n_mixtures, 0.3))
    components = rng.choice(n_mixtures, size=n_samples, p=weights)
    L = 0.05 + rng.gamma(shape=1.0, scale=0.5, size=(n_samples, n_mixtures))
    L[np.arange(n_samples), components] += rng.gamma(
        shape=6.0,
        scale=1.0,
        size=n_samples,
    )
    return L


def _load_reference_likelihood_matrix():
    _clear_invalid_r_home()
    from rpy2 import robjects

    data_path = fetch_daarem_mixdata()
    robjects.r(f'load("{data_path}")')
    return np.asarray(robjects.r("L"), dtype=np.float64)


def _project_simplex(x):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    total = x.sum()
    if total == 0:
        return np.full_like(x, 1.0 / x.size)
    return x / total


def _mixem_update(L, x, eps=1e-15):
    row_eps = eps * L.max(axis=1)
    posterior = L * x + row_eps[:, None]
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    return posterior.mean(axis=0)


def _mixobjective(L, x, eps=1e-15):
    return np.log(L @ x + eps * L.max(axis=1)).sum()


def _fit_em(L, x0, n_iter):
    x = x0.copy()
    values = []
    for _ in range(n_iter):
        x = _mixem_update(L, x)
        values.append(_mixobjective(L, x))
    return x, np.asarray(values)


def _fit_daarem(L, x0, n_iter, order=5):
    accelerator = AndersonEMAccelerator(order=order, monotone=True)
    x = _mixem_update(L, x0)
    values = [_mixobjective(L, x)]
    accelerator.update(
        x=torch.as_tensor(x0, dtype=torch.float64),
        g=torch.as_tensor(x, dtype=torch.float64),
    )

    for _ in range(1, n_iter):
        plain = _mixem_update(L, _project_simplex(x))
        plain_value = _mixobjective(L, plain)
        accelerator.update(
            x=torch.as_tensor(_project_simplex(x), dtype=torch.float64),
            g=torch.as_tensor(plain, dtype=torch.float64),
        )
        proposal = accelerator.propose()
        if proposal is not None:
            candidate = _project_simplex(proposal.candidate.numpy())
            candidate_value = _mixobjective(L, candidate)
            if candidate_value >= plain_value:
                x = candidate
                values.append(candidate_value)
                accelerator.accept()
                accelerator.update_cycle_monotonicity(
                    loglik=candidate_value,
                    history=proposal.history,
                )
                continue
            accelerator.update_cycle_monotonicity(
                loglik=plain_value,
                history=proposal.history,
            )
            accelerator.reject()
        x = plain
        values.append(plain_value)

    return _project_simplex(x), np.asarray(values)


def test_daarem_accelerates_mixture_proportion_em():
    L = _load_reference_likelihood_matrix()
    x0 = np.full(L.shape[1], 1.0 / L.shape[1])

    em_weights, em_values = _fit_em(L, x0, n_iter=50)
    daarem_weights, daarem_values = _fit_daarem(L, x0, n_iter=50, order=5)

    assert np.allclose(em_weights.sum(), 1.0)
    assert np.allclose(daarem_weights.sum(), 1.0)
    assert daarem_values[-1] > em_values[-1]
