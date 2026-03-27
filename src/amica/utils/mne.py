"""Helpers to export to MNE objects."""
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import mne

from .imports import import_optional_dependency


def to_mne(transformer, info: "mne.Info"):
    """Convert a fitted AMICA instance to an MNE-Python ICA instance.

    Parameters
    ----------
    transformer : amica.AMICA
        The fitted AMICA instance.
    info : instance of mne.Info
        Channel metadata with the same number of channels as the fitted AMICA
        instance.

    Returns
    -------
    ica : mne.preprocessing.ICA
        the MNE-Python ICA instance.
    """
    mne = import_optional_dependency("mne")
    check_is_fitted(transformer)

    n_observations = transformer.n_features_in_
    n_components = transformer.components_.shape[0]

    ica = mne.preprocessing.ICA(method="imported_eeglab", n_components=n_components)

    if not isinstance(info, mne.Info):
        raise TypeError(
            f"info must be an instance of mne.Info, got {type(info)!r}"
            )  # pragma: no cover

    ch_names = info["ch_names"]
    if len(ch_names) != n_observations:
        raise ValueError(
            f"The number of features in this AMICA instance ( {n_observations} ) "
            "does not match the number of channels in the info object: "
            f"( {len(ch_names)} )."
        )  # pragma: no cover

    # Borrowed from mne.read_ica_eeglab
    use = transformer._unmixing @ transformer.whitening_
    u, s, v = mne.fixes._safe_svd(use, full_matrices=False)

    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica.n_pca_components = None
    ica.pre_whitener_ = np.ones((n_observations, 1))
    ica.n_components_ = n_components
    if hasattr(transformer, "mean_") and transformer.mean_ is not None:
        ica.pca_mean_ = transformer.mean_.astype(np.float64, copy=True)
    else:
        ica.pca_mean_ = np.zeros(n_observations, dtype=np.float64)  # pragma: no cover

    ica.current_fit = "raw"
    ica.ch_names = ch_names
    ica.info = info
    ica.n_iter_ = transformer.n_iter_
    ica.exclude = []
    ica.reject_ = None

    ica._update_mixing_matrix()
    ica._update_ica_names()
    return ica
