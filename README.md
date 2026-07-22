[![codecov](https://codecov.io/github/scott-huberty/amica-python/graph/badge.svg?token=Gt7dvyE9mL)](https://codecov.io/github/scott-huberty/amica-python)
[![tests](https://github.com/scott-huberty/amica-python/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/scott-huberty/amica-python/actions/workflows/ci.yaml)
[![docs](https://img.shields.io/github/actions/workflow/status/scott-huberty/amica-python/circleci_redirect.yml?label=Docs)](https://dl.circleci.com/status-badge/redirect/gh/scott-huberty/amica-python/tree/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# AMICA-Python

A Python implementation of the [AMICA](https://sccn.ucsd.edu/~jason/amica_a.pdf) (Adaptive Mixture Independent Component Analysis) algorithm for blind source separation, that was originally [developed in FORTRAN](https://github.com/sccn/amica) by Jason Palmer at the Swartz Center for Computational Neuroscience (SCCN). AMICA-Python is correctness tested against the Fortran implementation.

| Python | Fortran |
|--------|---------|
| <img src="https://raw.githubusercontent.com/scott-huberty/amica-python/main/docs/source/_static/amica-python.gif" width=400px /> | <img src="https://raw.githubusercontent.com/scott-huberty/amica-python/main/docs/source/_static/amica-fortran.gif" width=400px /> |


## Installation

AMICA-Python is available from PyPI and conda-forge. Since AMICA-Python uses
PyTorch for the core numerical routines, installation depending on which PyTorch build
you need (e.g. CPU-only or with GPU support):

```bash
uv pip install "amica-python[torch-cpu]"
```

For a CUDA-enabled PyTorch install, use:

```bash
uv pip install "amica-python[torch-cuda]"
```

You can also install with pip:

```bash
python -m pip install "amica-python[torch-cpu]"
```

Or with conda-forge, which installs the PyTorch runtime dependency as part of
the AMICA-Python package:

```bash
conda install -c conda-forge amica-python
```

If you need a specific PyTorch build, install PyTorch first using the
instructions for your platform at [pytorch.org](https://pytorch.org/get-started/locally/),
then install AMICA-Python without the PyTorch extra:

```bash
python -m pip install amica-python
```

## Usage

AMICA-Python exposes a scikit-learn interface. Here is an example of how to use it:

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

<img src="https://scott-huberty.github.io/amica-python/_images/sphx_glr_plot_ica_blind_source_separation_001.png" alt="AMICA-Python vs FastICA outputs" width="50%" style="display: block; margin: 0 auto;"/>

### GPU acceleration

If PyTorch was installed with CUDA support, you can fit AMICA on GPU:

```python
ica = AMICA(device='cuda', random_state=0)
```

<br/>

For more examples and documentation, please see the [documentation](https://scott-huberty.github.io/amica-python/).