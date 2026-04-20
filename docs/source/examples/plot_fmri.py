"""
Deriving spatial maps from group fMRI data using ICA
====================================================

Various approaches exist to derive spatial maps or networks from
group fMRI data. The methods extract distributed brain regions that
exhibit similar BOLD fluctuations over time.

This example applies ICA to fMRI data measured while children
and young adults watch movies. The resulting maps are visualized using
Nilearn's atlas plotting tools.

This example is borrowed from the `nilearn Python package <https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_compare_decomposition.html>`_.
We visually compare AMICA to the CanICA method implemented in Nilearn, which is
specifically designed for group-level analysis of fMRI data. To keep the AMICA
path comparable to CanICA, we first build the same group matrix used by the
CanICA pipeline, then fit AMICA on that matrix, and finally project the learned
maps back into brain space for plotting.

.. tip::
    TL;DR - I would probably just use CanICA or some other Group ICA method
    when working with study-wide fMRI data. This tutorial is for demonstration purposes
    and does not advocate for the use of AMICA over other methods tailored for fMRI.
"""

from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import randomized_svd

from amica import AMICA
from nilearn.datasets import fetch_development_fmri
from nilearn.decomposition import CanICA
from nilearn.decomposition._base import _mask_and_reduce
from nilearn.image import iter_img, math_img
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show


def _orient_components(components):
    """Flip each component so its largest-magnitude peak is positive."""
    components = components.copy()
    for component in components:
        if abs(component.min()) > abs(component.max()):
            component *= -1.0
    return components


def fit_amica_components(canica, func_filenames):
    """Fit AMICA on the CanICA group matrix and return maps in brain space."""
    reduced_data = _mask_and_reduce(
        canica.masker_,
        func_filenames,
        n_components=canica.n_components,
        random_state=canica.random_state,
        n_jobs=canica.n_jobs,
    )
    group_pca_components, _, _ = randomized_svd(
        reduced_data.T,
        n_components=canica.n_components,
        transpose=True,
        random_state=canica.random_state,
        n_iter=3,
    )
    group_matrix = group_pca_components.T
    amica = AMICA(
        n_components=canica.n_components,
        batch_size=4096,
        max_iter=1000,
        tol=1e-6,
        random_state=canica.random_state,
        verbose=1,
    )
    spatial_maps = _orient_components(amica.fit_transform(group_matrix.T).T)
    components_img = canica.masker_.inverse_transform(spatial_maps)
    return amica, group_matrix, components_img


def plot_components(components_img, *, prefix, n_plots=6):
    """Plot the first few component maps with a common color scale."""
    vmax = np.percentile(np.abs(components_img.get_fdata()), 99.5)
    for i, cur_img in enumerate(iter_img(components_img)):
        if i >= n_plots:
            break
        plot_stat_map(
            cur_img,
            display_mode="z",
            title=f"{prefix} IC {i}",
            cut_coords=1,
            vmin=-vmax,
            vmax=vmax,
            colorbar=False,
        )


# %%
# Load brain development fMRI dataset
# -----------------------------------
rest_dataset = fetch_development_fmri(n_subjects=30)
func_filenames = rest_dataset.func

print(f"First functional nifti image (4D) is at: {rest_dataset.func[0]}")


# %%
# Apply ICA on the data
# ---------------------
output_dir = Path.cwd() / "results" / "plot_compare_decomposition"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")


# %%
# CanICA
# ^^^^^^
canica = CanICA(
    n_components=20,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
    random_state=0,
    mask_strategy="whole-brain-template",
    n_jobs=2,
)
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
    canica.fit(func_filenames)

canica_components_img = canica.components_img_
canica_components_img.to_filename(output_dir / "canica_resting_state.nii.gz")


# %%
# AMICA on the CanICA group matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_amica, amica_group_matrix, amica_components_img = fit_amica_components(
    canica,
    func_filenames,
)
amica_components_img.to_filename(output_dir / "amica_resting_state.nii.gz")
print(f"AMICA group matrix shape: {amica_group_matrix.shape}")


# %%
# Plot all components together
# ----------------------------
plot_prob_atlas(canica_components_img, title="All CanICA components")
plot_prob_atlas(
    math_img("np.abs(img)", img=amica_components_img),
    title="All AMICA components",
)


# %%
# Plot the first few component maps separately
# --------------------------------------------
plot_components(canica_components_img, prefix="CanICA")
plot_components(amica_components_img, prefix="AMICA")

show()

# %%
import shutil
shutil.rmtree(output_dir)

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=8
