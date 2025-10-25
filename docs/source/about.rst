About
=====


A Python implementation of the `AMICA <https://sccn.ucsd.edu/~jason/amica_a.pdf>`_
(Adaptive Mixture Independent Component Analysis) algorithm for blind source separation,
that was originally `developed in FORTRAN <https://github.com/sccn/amica>`_ by Jason
Palmer at the Swartz Center for Computational Neuroscience (SCCN).

AMICA-Python is pre-alpha but is tested and ready for test driving.

What is AMICA?
^^^^^^^^^^^^^^

Like the Infomax ICA algorithm, AMICA can identify sub and super-Gaussian sources. However,
AMICA goes a step further by modeling the components as arising from a mixture of multiple
source distributions. This design decision was likely motivated by the hypothesis that
neural activity, which can be recorded with electrophysiological (EEG), derives from
multiple source generators.


What does AMICA-Python implement?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- AMICA-Python implements what I consider to be the core AMICA algorithm, powered by
  `Torch <https://pytorch.org/>`_ and wrapped in an easy-to-use `scikit-learn <https://scikit-learn.org/stable/>`_ style interface.

- The outputs are numerically tested against the original FORTRAN implementation to ensure correctness and minimize bugs.


What wasn't implemented?
^^^^^^^^^^^^^^^^^^^^^^^^

- AMICA-Python does *not* implement all the features of the original FORTRAN implementation, Notably:

    - Run multiple models (i.e. run multiple decompositions) to fit different segments of data.

    - Reject samples based on a thresholded log-likelihood.

    - AMICA-Python does not expose all the parameters of the original AMICA algorithm.


