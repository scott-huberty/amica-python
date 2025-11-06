from __future__ import annotations

from collections.abc import Iterator
from typing import Union
from warnings import warn

import numpy as np
import psutil
import torch

ArrayLike2D = Union[np.ndarray, "np.typing.NDArray[np.floating]"]


class BatchLoader:
    """Iterate over an array in fixed-size batches of data along a chosen axis.

    We hand rolled this instead of using DataLoader because 1) we want to yield
    slices of input array (i.e. a view), and 2) return the indices as
    a slice object. DataLoader would internally convert the slice into a tensor
    of indices.

    Example (AMICA shape):
        X: (n_samples, n_features)
        it = BatchLoader(X, axis=0, batch_size=4096)
        for X_blk, sl in it:
            # X_blk is X[sl, :] where sl is slice(start, end)
            ...
    """

    def __init__(self, X: ArrayLike2D, axis: int, batch_size: int | None = None):
        # Validate inputs
        cls_name = self.__class__.__name__
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"{cls_name} expects a torch.Tensor")  # pragma: no cover
        if X.ndim < 1:
            raise ValueError(
                f"{cls_name} expects an array with at least 1 dimension"
            )  # pragma: no cover
        self.X = X
        self.axis = axis

        if self.axis < 0:
            self.axis += X.ndim
        if not (0 <= self.axis < X.ndim):
            raise ValueError(
                f"axis {self.axis} out of bounds for array with ndim={X.ndim}"
                )
        
        # Determine batching parameters
        n = X.shape[self.axis]
        start = 0
        stop = n
        if batch_size is None:
            # Treat as single chunk spanning [start:stop]
            batch_size = stop

        # Validate parameters
        assert (0 <= start <= n), f"start {start} out of range [0, {n}]"
        assert (0 <= stop <= n), f"stop {stop} out of range [0, {n}]"
        assert start <= stop, f"start {start} must be <= stop {stop}"        
        if batch_size < 0:
            raise ValueError(f"batch_size must be positive. Got {batch_size}.")
        if batch_size > X.shape[self.axis]:
            raise ValueError(
                f"batch_size {batch_size} exceeds data size {X.shape[self.axis]} "
                f"along axis {self.axis}."
            )

        # Store parameters
        self.start = start
        self.stop = stop
        self.batch_size = int(batch_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.start + idx * self.batch_size
        stop = min(start + self.batch_size, self.stop)


        idx = [slice(None)] * self.X.ndim
        idx[self.axis] = slice(start, stop)
        return self.X[tuple(idx)]

    def __iter__(self) -> Iterator[tuple[np.ndarray, slice]]:
        axis = self.axis
        start = self.start
        stop = self.stop
        step = self.batch_size

        idx = [slice(None)] * self.X.ndim
        assert -((stop - start) // -step) == len(self)  # sanity check
        for s in range(start, stop, step):
            e = min(s + step, stop)
            batch_slice = slice(s, e)
            idx[axis] = batch_slice
            yield self.X[tuple(idx)], batch_slice

    def __len__(self) -> int:
        return (self.X.shape[self.axis] + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(Data shape: {self.X.shape}, "
            f"Batched axis: {self.axis}, batch_size: {self.batch_size}, "
            f"n_batches: {len(self)})"
        )

def choose_batch_size(
        *,
        N: int,
        n_comps: int,
        n_mix: int,
        n_models: int = 1,
        dtype: np.dtype = np.float64,
        memory_fraction: float = 0.25,      # use up to 25% of available memory
        memory_cap: float = 1.5 * 1024**3,  # 1.5 GB absolute ceiling
        ) -> int:
    """
    Choose batch size for processing data in chunks.

    Parameters
    ----------
    N : int
        Total number of samples.
    n_comps : int
        Number of components to be learned in the model, e.g. size of the n_components
        dimension of the data.
    n_mix : int
        Number of mixture components per source/component to be learned in the model.
    dtype : np.dtype, optional
        Data type of the input data, by default np.float64.
    memory_cap : float, optional
        Maximum memory (in bytes) to be used for processing, by default
        ``1.5 * 1024**3`` (1.5 GB).

    Notes
    -----
    The batch size is primarily determined by the estimated size of hot buffers (e.g.
    y, z, fp, ufp), which scale with the size of n_samples:
    - One array of shape (N,):
        - loglik
    - Two arrays of shape (N, n_models):
        - modloglik
        - v (model responsibilities)
    - Two arrays of shape (N, n_comps)
        - b
        - g
    - Five arrays of shape (N, n_comps, n_mix): u, y, z, fp, ufp
        - u (mixture responsibilities)
        - y
        - z
        - fp
        - ufp
    """
    dtype_size = np.dtype(dtype).itemsize
    # per-sample cost across pre-allocated buffers
    bytes_per_sample = (
        1                       # loglik
        + 2 * n_models          # modloglik, v
        + 2 * n_comps           # b, g
        + 5 * n_comps * n_mix   # fp, u, ufp, y, z,
        ) * dtype_size
    # Plus small headroom for intermediates
    bytes_per_sample = int(bytes_per_sample * 1.2)

    # Pick memory budget
    try:
        hard_cap = 4 * 1024**3  # 4 GiB (avoid runaway memory use)
        avail_mem = psutil.virtual_memory().available
        mem_cap = min(avail_mem * memory_fraction, hard_cap)
    except Exception:
        mem_cap = memory_cap  # fallback to user-specified cap

    max_batch_size = mem_cap // bytes_per_sample

    # Ensure at least 1 sample. This should only trigger if n_comps and n_mix are huge.
    if max_batch_size < 1:
        raise MemoryError(
            f"Cannot fit even 1 sample within memory cap of "
            f"{mem_cap / 1024**3:.2f} GiB. "
            f"Per-sample memory cost is {bytes_per_sample / 1024**3:.2f} GB."
        )
    batch_size = int(min(N, max_batch_size))

    # Heuristic floor, we don't want absurdly small chunks or chunks that are too
    # small relative to the model complexity (n_comps)
    # This heuristic works well for typical ICA regimes, where n_comps is < 256
    min_batch_size = max(8192, n_comps * 32)  # at least 32 samples per component
    min_batch_size = min(min_batch_size, N)  # Cannot exceed N
    if batch_size < min_batch_size:
        warn(
            f"Warning: To stay within the memory cap, batch size is {batch_size} "
            f"samples, which is below the recommended minimum of {min_batch_size}."
        )
    return batch_size


def get_component_slice(h: int, n_components: int) -> slice:
    """Return slice for components of model h.

    Parameters
    ----------
    - h: model number (1-based)
    - n_components: number of components per model (nw)

    Returns
    -------
    - slice object for components of model h

    Notes
    -----
    - Creating a slice ensures that we get a view when indexing arrays.
    - The fortran code pre-computes comp_list(num_components, num_models). We avoid this
        by computing the slice on-the-fly and thus avoiding fancy indexing.

    Fortran reference:
        do h = 1,num_models
            do i = 1,nw
                comp_list(i,h) = (h-1) * nw + i
    """
    h_index = h - 1  # Convert to 0-based index
    start = h_index * n_components
    end = start + n_components
    return slice(start, end)
