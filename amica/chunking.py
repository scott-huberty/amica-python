from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterator, Optional, Sequence, Tuple, Union

import numpy as np


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
        if not isinstance(X, np.ndarray):
            raise TypeError("ChunkIterator expects a numpy ndarray")
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
        for s in range(start, stop, step):
            e = min(s + step, stop)
            blk_slice = slice(s, e)
            idx[axis] = blk_slice
            yield self.X[tuple(idx)], blk_slice

