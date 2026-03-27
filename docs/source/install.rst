
Installation
============

For now, AMICA-Python should be installed from source, and you will have to manually install
PyTorch (see below) yourself.

Clone the repository, and install it using pip:

.. code-block:: bash

    git clone https://github.com/scott-huberty/amica-python.git
    cd amica-python
    pip install -e .

.. Important::

   AMICA-Python requires PyTorch. Please install PyTorch
   by following the instructions at `pytorch.org <https://pytorch.org/get-started/locally/>`_
   before using AMICA-Python.

Installing PyTorch
~~~~~~~~~~~~~~~~~~

To install PyTorch, please follow the instructions at `pytorch.org <https://pytorch.org/get-started/locally/>`_:


.. tab-set::

   .. tab-item:: Pip

      .. code-block:: bash

         pip install torch

   .. tab-item:: Conda

      .. code-block:: bash

         conda install -c conda-forge pytorch


If you wish, you can install the CPU-only version of PyTorch:


.. tab-set::

   .. tab-item:: Pip

      .. code-block:: bash

         pip install torch --index-url https://download.pytorch.org/whl/cpu

   .. tab-item:: Conda

      .. code-block:: bash

         conda install -c conda-forge pytorch cpuonly


If you use UV, you can install Pytorch at the same time as AMICA-Python:

.. tab-set::

   .. tab-item:: Torch with GPU support

      .. code-block:: bash

         uv pip install -e ".[torch-cuda]"

   .. tab-item:: CPU-only Torch

      .. code-block:: bash

         uv pip install -e ".[torch-cpu]"


GPU-accelerated
~~~~~~~~~~~~~~~

Once PyTorch with CUDA support is installed, you can run AMICA on GPU:

.. code-block:: python

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

   ica = AMICA(random_state=0, device="cuda")  # default is "cpu"
   X_new = ica.fit_transform(X)
