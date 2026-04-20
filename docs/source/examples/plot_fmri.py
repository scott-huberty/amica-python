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
path lightweight, we first reduce the subject data to a shared PCA space, fit
AMICA in that reduced space, and then project the learned maps back into brain
space for plotting.

The reference paper for CanICA is :footcite:t:`Varoquaux2010c`.
"""

from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.exceptions import ConvergenceWarning

from amica import AMICA
from nilearn.datasets import fetch_development_fmri
from nilearn.decomposition import CanICA
from nilearn.image import iter_img, math_img
from nilearn.maskers import MultiNiftiMasker
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show


def _orient_components(components):
    """Flip each component so its largest-magnitude peak is positive."""
    components = components.copy()
    for component in components:
        if abs(component.min()) > abs(component.max()):
            component *= -1.0
    return components


def fit_amica_components(func_filenames, *, n_components=20, random_state=0):
    """Fit AMICA on PCA-reduced fMRI signals and return maps in brain space."""
    masker = MultiNiftiMasker(
        smoothing_fwhm=6,
        standardize=True,
        detrend=True,
        mask_strategy="whole-brain-template",
        memory="nilearn_cache",
        memory_level=1,
        n_jobs=2,
    )
    masker.fit(func_filenames)

    pca = IncrementalPCA(n_components=n_components)
    for func_file in func_filenames:
        pca.partial_fit(masker.transform(func_file))

    reduced_data = np.vstack(
        [pca.transform(masker.transform(func_file)) for func_file in func_filenames]
    )

    amica = AMICA(
        n_components=n_components,
        batch_size=4096,
        max_iter=1000,
        tol=1e-6,
        random_state=random_state,
        verbose=1,
    )
    amica.fit(reduced_data)

    components = _orient_components(amica.mixing_.T @ pca.components_)
    components_img = masker.inverse_transform(components)
    return amica, components_img


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
            #vmin=-vmax,
            #vmax=vmax,
            colorbar=False,
        )


# %%
# Load brain development :term:`fMRI` dataset
# -------------------------------------------
rest_dataset = fetch_development_fmri(n_subjects=30)
func_filenames = rest_dataset.func

print(f"First functional nifti image (4D) is at: {rest_dataset.func[0]}")


# %%
# Apply ICA on the data
# ---------------------
output_dir = Path(__file__).parent / "results" / "plot_compare_decomposition"
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
# AMICA
# ^^^^^
_amica, amica_components_img = fit_amica_components(
    func_filenames,
    n_components=20,
    random_state=0,
)
amica_components_img.to_filename(output_dir / "amica_resting_state.nii.gz")


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
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=8
