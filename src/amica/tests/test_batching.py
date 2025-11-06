import pytest
import torch

from amica._batching import BatchLoader


def test_batch_loader():
    """Test accessing input data in batches using BatchLoader."""
    # Vector of 1000 1s, 1000 2s, 1000 3s
    X = torch.concatenate(
        [torch.ones((1000, 2)), 2 * torch.ones((1000, 2)), 3 * torch.ones((1000, 2))]
        )
    batch_loader = BatchLoader(X, axis=0, batch_size=1000)

    # There should be 3 batches
    assert len(batch_loader) == 3
    # Test __getitem__
    assert torch.all(batch_loader[0] == 1)
    assert torch.all(batch_loader[1] == 2)
    assert torch.all(batch_loader[2] == 3)
    # Test __iter__
    for i, (batch, batch_slice) in enumerate(batch_loader):
        expected_value = i + 1
        assert torch.all(batch == expected_value)
        assert batch_slice == slice(i * 1000, (i + 1) * 1000)
    # Test __repr__
    repr_str = repr(batch_loader)
    assert "Data shape: torch.Size([3000, 2])" in repr_str
    assert "Batched axis: 0" in repr_str
    assert "batch_size: 1000" in repr_str
    assert "n_batches: 3" in repr_str

    # Test batch size None
    foo = BatchLoader(X, axis=0, batch_size=None)
    assert len(foo) == 1
    assert foo[0].shape == X.shape

    # Test failures
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchLoader(X, axis=0, batch_size=-10)
    with pytest.raises(ValueError, match="batch_size 4000 exceeds data size 3000"):
        BatchLoader(X, axis=0, batch_size=4000)

