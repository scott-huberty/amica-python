About
=====


A Python implementation of the `AMICA <https://sccn.ucsd.edu/~jason/amica_a.pdf>`_
(Adaptive Mixture Independent Component Analysis) algorithm for blind source separation,
that was originally `developed in FORTRAN <https://github.com/sccn/amica>`_ by Jason
Palmer at the Swartz Center for Computational Neuroscience (SCCN).

What is AMICA?
^^^^^^^^^^^^^^

Like the Infomax ICA algorithm, AMICA can identify sub and super-Gaussian sources. However,
AMICA goes a step further by modeling the sources themselves as a mixture of multiple
Generalized Gaussian distributions:

.. image:: _static/GGD.png
   :alt: Source distribution modeled as a generalized Gaussian mixture.
   :width: 70%
   :align: center

