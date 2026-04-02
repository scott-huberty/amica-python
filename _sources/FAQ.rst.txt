FAQ
===


How did AMICA-Python choose its default parameter values?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AMICA-Python uses the default parameter values from the MATLAB wrapper, not the FORTRAN
headers or the standalone parameter file for the test data. We decided the MATLAB 
interface has historically been the primary user-facing entry point for AMICA.

Will AMICA-Python support fitting multiple ICA decompositions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original FORTRAN AMICA implementation can fit multiple ICA decompositions
(``num_models > 1``). AMICA-Python is intentionally scoped to the single-model case:

1. Most users only want a single decomposition.
2. AMICA is already computationally expensive in terms of memory use, runtime, and
   convergence behavior. Fitting multiple models increases this cost.
3. A one-model scope is the right fit for the :class:`amica.AMICA` estimator API:
   it aligns with scikit-learn ergonomics and keeps initialization,
   reproducibility (seeds), and parameter access (for example loading and
   inspecting weights) straightforward.

If future demand for multiple models is strong, it would likely be exposed as
a higher-level meta-estimator that accepts multiple AMICA instances, and selects the
dominant model per sample and transforms the data accordingly.

Why doesn't AMICA-Python implement sample rejection by low log-likelihood?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is primarily a pragmatic design decision. Masking samples during optimization
typically requires fancy indexing that materializes temporary array copies. For
AMICA-scale tensors, especially those that grow with ``n_samples`` and ``n_mixtures``,
this can increase both memory usage and runtime.

Sample rejection also introduces parameter complexity and tuning burden, e.g. threshold
choice, when to start rejection, and related scheduling. If configured poorly, rejection
can remove samples that would later fit well, destabilize training, or collapse
solutions.

AMICA-Python aims for a focused, robustly tested scope that avoids excessive
branching logic and feature surface area.