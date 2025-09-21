from pathlib import Path

import pytest
import numpy as np
from numpy.testing import assert_allclose
import torch

from amica.kernels import (
    compute_preactivations,
    compute_source_densities,
    compute_model_loglikelihood_per_sample,
    compute_total_loglikelihood_per_sample,
    compute_model_responsibilities,
    compute_mixture_responsibilities,
    compute_weighted_responsibilities,
    compute_source_scores,
    precompute_weighted_scores,
    compute_scaled_scores,
)


# Data for testing
torch.set_default_dtype(torch.float64)

# compute_preactivations
data_dir = Path(__file__).parent / "data"
data_batch = torch.from_numpy(np.load(data_dir / "data_batch.npy"))
wc = torch.from_numpy(np.load(data_dir / "wc.npy"))
W = torch.from_numpy(np.load(data_dir / "W.npy"))

# compute_source_densities
b = torch.from_numpy(np.load(data_dir / "b.npy"))
alpha = torch.from_numpy(np.load(data_dir / "alpha.npy"))
sbeta = torch.from_numpy(np.load(data_dir / "sbeta.npy"))
mu = torch.from_numpy(np.load(data_dir / "mu.npy"))
rho = torch.from_numpy(np.load(data_dir / "rho.npy"))

# compute_model_loglikelihood_per_sample
modloglik_init = torch.full((93, 1), -65.9306)
z0 = torch.from_numpy(np.load(data_dir / "z0.npy"))

# compute_total_loglikelihood_per_sample
modloglik = torch.from_numpy(np.load(data_dir / "modloglik.npy"))
loglik_init = torch.zeros(data_batch.shape[0])

# compute_weighted_responsibilities
v = torch.ones((93, 1))
z = torch.from_numpy(np.load(data_dir / "z.npy"))

# compute_source_scores
y = torch.from_numpy(np.load(data_dir / "y.npy"))

# compute_source_scores
fp = torch.from_numpy(np.load(data_dir / "fp.npy"))

# accumulate_scores
u = torch.from_numpy(np.load(data_dir / "u.npy"))


@pytest.mark.parametrize("data_batch, wc, W", [(data_batch, wc, W)])
def test_loaded_data(data_batch, wc, W):
    """Test that the loaded data is correct."""
    assert data_batch.size() == (93, 32)
    assert wc.size() == (32, 1)
    assert W.size() == (32, 32, 1)
    assert_allclose(data_batch[:3, 0], [-0.18746214, -0.15889934, -0.05030178])
    assert_allclose(W.numpy()[0, :2, 0], [1.0000898174008426, -0.0032845277326563212])
    assert_allclose(wc, 0)


@pytest.mark.parametrize("data_batch, wc, W", [(data_batch, wc, W)])
def test_compute_preactivations(data_batch, wc, W):
    """Test the compute_preactivations function."""
    b = compute_preactivations(
        X=data_batch,
        unmixing_matrix=W[:, :, 0],
        bias=wc[:, 0],
    )
    assert b.size() == data_batch.size()
    assert_allclose(b[0, :2], [-0.18617958842145835,  0.43923809859505])


@pytest.mark.parametrize("b, alpha, sbeta, mu, rho", [(b, alpha, sbeta, mu, rho)])
def test_compute_source_densities(b, alpha, sbeta, mu, rho):
    """Test the compute_source_densities function."""
    log_p, responsibilities = compute_source_densities(
        b=b,
        alpha=alpha,
        sbeta=sbeta,
        mu=mu,
        rho=rho,
        pdftype=0,
        comp_slice=slice(None, None),
    )
    assert log_p.size() == (93, 32, 3)
    assert responsibilities.size() == (93, 32, 3)
    assert_allclose(log_p[:2, 0, 0], [0.7865425191927538, 0.806627002546346])
    assert_allclose(responsibilities[-1:, -1, -1], [-1.98167829451754])


def test_compute_mixture_responsibilities():
    """Test the compute_mixture_responsibilities function."""
    z = compute_mixture_responsibilities(log_densities=z0, inplace=False)
    assert z.size() == (93, 32, 3)
    assert_allclose(z[:2, 0, 0], [0.29726705022373134, 0.2879118893720002])


@pytest.mark.parametrize("z0, modloglik", [(z0, modloglik_init)])
def test_compute_model_loglikelihood_per_sample(z0, modloglik):
    """Test the compute_model_loglikelihood_per_sample function."""
    mod_logits = compute_model_loglikelihood_per_sample(
        log_densities=z0,
        out_modloglik=modloglik[:, 0].clone(),
    )
    assert mod_logits.size() == (93,)
    assert_allclose(mod_logits[:2], [-117.4721353091377, -114.67331045267954])


@pytest.mark.parametrize("modloglik, loglik_init", [(modloglik, loglik_init)])
def test_compute_total_loglikelihood_per_sample(modloglik, loglik_init):
    """Test the compute_total_loglikelihood_per_sample function."""
    log_likelihood = compute_total_loglikelihood_per_sample(
        modloglik=modloglik,
        out_loglik=loglik_init.clone(),
    )
    assert log_likelihood.size() == (93,)
    assert_allclose(log_likelihood[:2], [-117.4721353091377, -114.67331045267954])


@pytest.mark.parametrize("modloglik", [(modloglik)])
def test_compute_model_responsibilities(modloglik):
    """Test the compute_model_responsibilities function."""
    v = compute_model_responsibilities(modloglik=modloglik)
    assert v.size() == (93, 1)
    assert_allclose(v, 1)
    assert_allclose(v[:, 0].sum(), 93)


@pytest.mark.parametrize("z, v", [(z, v)])
def test_compute_weighted_responsibilities(z, v):
    """Test the compute_weighted_responsibilities function."""
    u = compute_weighted_responsibilities(
        mixture_responsibilities=z,
        model_responsibilities=v[:, 0],
        single_model=True,
        )
    assert u.size() == (93, 32, 3)
    assert_allclose(u[:2, 0, 0], [0.29726705022373134, 0.2879118893720002])
    assert_allclose(u.sum(dim=0)[-1, -1], 44.685364917371935)


@pytest.mark.parametrize("y, rho", [(y, rho)])
def test_compute_source_scores(y, rho):
    """Test the compute_source_scores function."""
    fp = compute_source_scores(
        y=y, rho=rho, pdftype=0, comp_slice=slice(None, None)
        )
    assert fp.size() == (93, 32, 3)
    assert_allclose(fp[:2, 0, 0], [1.3303084860977532, 1.3471862364681724])


@pytest.mark.parametrize("u, fp", [(u, fp)])
def test_precompute_weighted_scores(u, fp):
    """Test the precompute_weighted_scores function."""
    ufp = precompute_weighted_scores(
        scores=fp,
        weighted_responsibilities=u,
        )
    assert ufp.size() == (93, 32, 3)
    assert_allclose(ufp[:2, 0, 0], [0.3954568795498768, 0.3878709346775057])


@pytest.mark.parametrize("u, fp, sbeta", [(u, fp, sbeta)])
def test_compute_scaled_scores(u, fp, sbeta):
    """Test the compute_scaled_scores function."""
    g = compute_scaled_scores(
        weighted_scores=u * fp,
        scales=sbeta, 
        )
    assert g.size() == (93, 32)
    assert_allclose(g[:2, 0], [-0.24500792120056003, -0.243892663692031])
