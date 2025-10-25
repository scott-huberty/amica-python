
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

