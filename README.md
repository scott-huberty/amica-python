[![codecov](https://codecov.io/github/scott-huberty/amica-python/graph/badge.svg?token=Gt7dvyE9mL)](https://codecov.io/github/scott-huberty/amica-python)
[![tests](https://github.com/scott-huberty/amica-python/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/scott-huberty/amica-python/actions/workflows/ci.yaml)
[![docs](https://img.shields.io/github/actions/workflow/status/scott-huberty/amica-python/circleci_redirect.yml?label=Docs)](https://dl.circleci.com/status-badge/redirect/gh/scott-huberty/amica-python/tree/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AMICA-Python

A Python implementation of the [AMICA](https://sccn.ucsd.edu/~jason/amica_a.pdf) (Adaptive Mixture Independent Component Analysis) algorithm for blind source separation, that was originally [developed in FORTRAN](https://github.com/sccn/amica) by Jason Palmer at the Swartz Center for Computational Neuroscience (SCCN).

AMICA-Python is pre-alpha but is tested and ready for test driving.

| Python | Fortran |
|--------|---------|
| <img src="https://raw.githubusercontent.com/scott-huberty/amica-python/main/docs/source/_static/amica-python.gif" width=400px /> | <img src="https://raw.githubusercontent.com/scott-huberty/amica-python/main/docs/source/_static/amica-fortran.gif" width=400px /> |

## What is AMICA?

Like the Infomax ICA algorithm, AMICA can identify sub and super-Gaussian sources. However, AMICA goes a step further by modeling the components as arising from a mixture of multiple source distributions. This design decision was likely motivated by the hypothesis that neural activity, which can be recorded with electrophysiological (EEG), derives from mutiple source generators.

# What does AMICA-Python implement?

- AMICA-Python implements what I consider to be the core AMICA algorithm, powered by [Torch](https://pytorch.org/) and wrapped in an easy-to-use [scikit-learn](https://scikit-learn.org/stable/) style interface.

- The outputs are numerically tested against the original FORTRAN implementation to ensure correctness and minimize bugs.

# What wasn't implemented?

- AMICA-Python does not implement all the features of the original FORTRAN implementation. Notably, it does not implement the following:

  - The ability to model multiple ICA decompositions simultaneously.
  - The ability to reject samples based on a thresholded log-likelihood.
  - AMICA-Python does not expose all the options available in the original FORTRAN implementation.

If you want these features, there is an alternative Python package called [pyAmica](https://github.com/neuromechanist/pyAMICA), though as of this writing, the author notes that pyAMICA is a work-in-progress and should not be used for research of production purposes.

## Installation

For now, AMICA-Python should be installed from source, and you will have to manually install
PyTorch (see below) yourself.

Please clone the repository, and install it using pip:

```bash
git clone https://github.com/scott-huberty/amica-python.git
cd amica-python
pip install -e .
```

> [!IMPORTANT]
> You must install PyTorch before using AMICA-Python.

### Installing PyTorch

Depending on your system and preferences, you can install PyTorch with or without GPU support. AMICA-Python actually does not yet support GPU acceleration, so you won't gain anything by installing the GPU version of PyTorch.


To install the standard version of PyTorch, run:

```bash
python -m pip install torch
```

>[!WARNING]
> If you are using an Intel Mac, you cannot install Pytorch via pip, because there are no precompiled wheels for that platform. Instead, you must install PyTorch via Conda, e.g.:

```bash
conda install pytorch -c conda-forge
```

To install the CPU-only version of PyTorch, run:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu113
```

Or for Conda users:

```bash
conda install -c conda-forge pytorch cpuonly
```

If you use UV, you can also just install torch while installing AMICA-Python:

```bash
uv pip install -e ".[torch-cpu]"
```

```bash
uv pip install -e ".[torch-cuda]"
```

## Usage

AMICA-Python exposes a scikit-learn style interface. Here is an example of how to use it:

```python
import numpy as np
from scipy import signal
from amica import AMICA


rng = np.random.default_rng(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)                     # Sinusoidal
s2 = np.sign(np.sin(3 * time))            # Square wave
s3 = signal.sawtooth(2 * np.pi * time)    # Sawtooth

S = np.c_[s1, s2, s3]
S += 0.2 * rng.standard_normal(S.shape)   # Add noise
S /= S.std(axis=0)                        # Standardize

A = np.array([[1, 1, 1],
              [0.5, 2, 1.0],
              [1.5, 1.0, 2.0]])           # Mixing matrix

X = S @ A.T                               # Observed mixtures

ica = AMICA(random_state=0)
X_new = ica.fit_transform(X)
```

For more examples and documentation, please see the [documentation site](https://scott-huberty.github.io/amica-python/).
