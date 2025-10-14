.. AMICA-Python documentation master file, created by
   sphinx-quickstart on Sun Sep 28 13:05:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AMICA-Python documentation
==========================

AMICA (Adaptive Mixture ICA) is a blind source separation algorithm for separating
multivariate signals. It is an extension of the Infomax algorithm, such that like
Infomax, AMICA can identify sources with sub and super gaussian distributions. It is an
extension of Infomax in that AMICA models sources as a linear mixture of multiple data
distributions. This design decision is owed to the fact that AMICA was first proposed by
neuroimaging researchers. It is widely assumed that brain waves recorded by MEG and EEG
devices are a mixture of multiple neuronal sources in the brain. AMICA allows this to be
explicitly modeled. However AMICA is not limited to neuorimaging data, and can be 
generally used for many BSS problems. See the examples below for a few use cases.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   auto_examples/index

