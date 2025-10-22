"""Type hints for AMICA arrays."""
from typing import Annotated, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

SamplesVector: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples,)"]
"""Alias for a 1D array with shape (n_samples,)."""

ComponentsVector: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_components,)"]
"""Alias for a 1D array with shape (n_components,)."""

DataArray2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_features, n_samples)"]
"""Alias for a 2D array with shape (n_features, n_samples)."""

DataTensor2D: TypeAlias = Annotated[torch.Tensor, "(n_samples, n_features)", 2]
"""Alias for a 2D Tensor with shape (n_samples, n_features)."""

WeightsArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_components, n_features)"]
"""Alias for a 2D array with shape (n_components, n_features)."""

SourceArray2D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_components)"]
"""Alias for a 2D array with shape (n_samples, n_components)."""

SourceArray3D: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_components, n_mixtures)"]
"""Alias for a 3D array with shape (n_samples, n_components, n_mixtures)."""

ParamsArray: TypeAlias  = Annotated[npt.NDArray[np.float64], "(n_components, n_mixtures)"]
"""Alias for a 2D array with shape (n_components, n_mixtures)."""

ParamsModelArray: TypeAlias  = (
    Annotated[npt.NDArray[np.float64], "(n_components, n_models)"]
)
"""Alias for a 2D array with shape (n_components, n_models)."""

ParamsModelTensor: TypeAlias  = Annotated[torch.Tensor, "(n_components, n_models)", 2]
"""Alias for a 2D Tensor with shape (n_components, n_models)."""

LikelihoodArray: TypeAlias = Annotated[npt.NDArray[np.float64], "(n_samples, n_models)"]
"""Alias for a 2D array with shape (n_samples, n_models)."""

ScalarTensor: TypeAlias = Annotated[torch.Tensor, "()", 0]
"""Alias for a 0D Tensor (scalar), i.e. with 1 element."""
