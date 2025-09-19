from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import psutil

ArrayLike2D = Union[np.ndarray, "np.typing.NDArray[np.floating]"]


class ChunkIterator:
    """Iterate over an array in fixed-size chunks along a chosen axis.

    This is intentionally simple and dependency-free. It yields slices of the
    input array without copying (views), suitable for streaming large inputs
    through algorithms that operate on blocks.

    Example (AMICA shape):
        X: (n_channels, n_samples)
        it = ChunkIterator(X, axis=1, chunk_size=4096)
        for X_blk, sl in it:
            # X_blk is X[:, sl] where sl is slice(start, end)
            ...
    """

    def __init__(self, X: ArrayLike2D, axis: int, chunk_size: Optional[int] = None):
        if not isinstance(X, torch.Tensor):
            raise TypeError("ChunkIterator expects a torch.Tensor")
        if X.ndim < 1:
            raise ValueError("ChunkIterator expects an array with at least 1 dimension")
        self.X = X
        self.axis = axis

        if self.axis < 0:
            self.axis += X.ndim
        if not (0 <= self.axis < X.ndim):
            raise ValueError(f"axis {self.axis} out of bounds for array with ndim={X.ndim}")

        n = X.shape[self.axis]
        start = 0
        stop = n
        if not (0 <= start <= n):
            raise ValueError(f"start {start} out of range [0, {n}]")
        if not (0 <= stop <= n):
            raise ValueError(f"stop {stop} out of range [0, {n}]")
        if start > stop:
            raise ValueError(f"start {start} must be <= stop {stop}")
        self.start = start
        self.stop = stop

        if chunk_size is None or chunk_size <= 0:
            # Treat as single chunk spanning [start:stop]
            self.chunk_size = stop
        else:
            self.chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, slice]]:
        axis = self.axis
        start = self.start
        stop = self.stop
        step = self.chunk_size

        # Handle empty span quickly
        if start == stop:
            return iter(())

        idx = [slice(None)] * self.X.ndim
        assert (stop - start) // step == len(self)  # sanity check
        for s in range(start, stop, step):
            e = min(s + step, stop)
            blk_slice = slice(s, e)
            idx[axis] = blk_slice
            yield self.X[tuple(idx)], blk_slice
    
    def __len__(self) -> int:
        return self.X.shape[self.axis] // self.chunk_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(Data shape: {self.X.shape}, "
            f"Chunked axis: {self.axis}, chunk_size: {self.chunk_size}, "
            f"n_chunks: {len(self)})"
        )

def choose_chunk_size(
        *,
        N: int,
        n_comps: int,
        n_mix: int,
        dtype: np.dtype = np.float64,
        memory_fraction: float = 0.25,      # use up to 25% of available memory
        memory_cap: float = 1.5 * 1024**3,  # 1.5 GB absolute ceiling
        ) -> int:
    """
    Choose chunk size for processing data in chunks.

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
    The chunk size is primarily determined by the estimated size of the pre-allocated hot
    buffers in AmicaWorkspace, which scale with the size of n_samples, n_comps, and n_mix:
    - Two arrays of shape (N, n_comps): b, scratch_2ds
    - Four arrays of shape (N, n_comps, n_mix): y, z, fp, ufp
    """
    dtype_size = np.dtype(dtype).itemsize
    # per-sample cost across pre-allocated buffers in AmicaWorkspace
    bytes_per_sample = (2 * n_comps + 4 * n_comps * n_mix) * dtype_size
    # Add small overhead headroom for temporary ops and Python/NumPy overhead
    bytes_per_sample = int(bytes_per_sample * 1.1)

    # Pick memory budget
    try:
        hard_cap = 4 * 1024**3  # 4 GiB to avoid runaway memory use
        avail_mem = psutil.virtual_memory().available
        mem_cap = min(avail_mem * memory_fraction, hard_cap)
    except Exception:
        mem_cap = memory_cap  # fallback to user-specified cap

    max_chunk_size = mem_cap // bytes_per_sample

    # Ensure at least 1 sample. This should only trigger if n_comps and n_mix are huge.
    if max_chunk_size < 1:
        raise MemoryError(
            f"Cannot fit even 1 sample within memory cap of "
            f"{mem_cap / 1024**3:.2f} GB. "
            f"Per-sample memory cost is {bytes_per_sample / 1024**3:.2f} GB."
        )
    chunk_size = int(min(N, max_chunk_size))

    # Heuristic floor, we don't want absurdly small chunks or chunks that are too
    # small relative to the model complexity (n_comps)
    # This heuristic works well for typical ICA regimes, where n_comps is < 256
    min_chunk_size = max(8192, n_comps * 32)  # at least 32 samples per component
    min_chunk_size = min(min_chunk_size, N)  # Cannot exceed N
    if chunk_size < min_chunk_size:
        print(
            f"Warning: To stay within the memory cap, chunk size is {chunk_size}, "
            f"which is below the recommended minimum of {min_chunk_size}."
        )
    return chunk_size
