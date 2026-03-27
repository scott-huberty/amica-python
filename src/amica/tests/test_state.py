import pytest
import torch

from amica.state import AmicaConfig, get_initial_state


@pytest.fixture
def state():
    """Make an AMICAState instance."""
    cfg = AmicaConfig(
        n_features=3,
        n_components=3,
        n_models=2,
        n_mixtures=2,
        batch_size=4,
    )
    out = get_initial_state(cfg)
    out.W.fill_(0.0)
    out.sbeta.fill_(1.0)
    return out


def test_to_device_input(state):
    """Test that we can pass in a torch.device."""
    state.to_device(torch.device("cpu"))
    for value in state.to_dict().values():
        assert value.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_move_to_cuda(state):
    """Test moving things to GPU."""
    state.to_device("cuda")
    for value in state.to_dict().values():
        assert value.device.type == "cuda"

    # and test moving back
    state.to_device("cpu")
    for value in state.to_dict().values():
        assert value.device.type == "cpu"
