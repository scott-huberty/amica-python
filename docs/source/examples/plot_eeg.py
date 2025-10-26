"""
Run AMICA on EEG Data
=====================

And compare results to Fortran AMICA.

"""
# %%
import amica
from amica import AMICA
import matplotlib.pyplot as plt
import mne

import numpy as np

# %%
# Download sample data
# ^^^^^^^^^^^^^^^^^^^^
#

# %%
data_path = amica.datasets.data_path()

# %%
# Load Fortran AMICA initial weights and results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This is not necessary to run AMICA in Python, but we are going to compare amica-python
# results to those obtained from the original Fortran implementation.
#

# %%
initial_weights, initial_scales, initial_locations = amica.utils.load_initial_weights(
    data_path / "eeglab_sample_data" / "amicaout_test", n_components=32, n_mixtures=3
    )

# %%
# Load EEG data
# ^^^^^^^^^^^^^
#

# %%
amica_outdir = data_path / "eeglab_sample_data" / "amicaout_test"
fortran_results = amica.utils.load_fortran_results(
    amica_outdir, n_components=32, n_mixtures=3
    )

# %%
raw = mne.io.read_raw_eeglab(
    data_path / "eeglab_sample_data"/ "eeglab_data.set", preload=True
    )
data = raw.get_data().T  # Shape (n_samples, n_channels)
data *= 1e6  # Convert from Volts to microVolts


# %%
# Run AMICA-Python
# ^^^^^^^^^^^^^^^^^^^^^
#

# %%
transformer = AMICA(
        max_iter=200,
        w_init=initial_weights,
        sbeta_init=initial_scales,
        mu_init=initial_locations,
)
transformer.fit(data)

# %%
# Compare results
# ^^^^^^^^^^^^^^^
#

# %%
def plot_topomaps(A, output="python"):
    fig, ax = plt.subplots(
        nrows=8,
        ncols=4,
        figsize=(8, 12),
        constrained_layout=True
        )
    for i, this_ax in zip(range(32), ax.flat):
        mne.viz.plot_topomap(
            A[:, i],
            pos=raw.info,
            axes=this_ax,
            show=False,
        )
        this_ax.set_title(f"Component {i}")
    fig.suptitle(f"AMICA Component Topomaps ({output})", fontsize=16)
    return fig, ax

# %%
fig1, ax1 = plot_topomaps(transformer.mixing_, output="python")


# %%
# The Fortran mixing matrix is in sphered space. We need to unwhiten it first.
A_fortran = np.linalg.pinv(fortran_results['S']) @ fortran_results['A']
fig2, ax2 = plot_topomaps(A_fortran, output="fortran")