"""
Basic usage of amica
======================
"""
# %%
import amica
import matplotlib.pyplot as plt
import mne

# %%
data_path = amica.datasets.data_path()

# %%
initial_weights, initial_scales, initial_locations = amica.utils.load_initial_weights(
    data_path / "eeglab_sample_data" / "amicaout_test", n_components=32, n_mixtures=3
    )

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
results = amica.fit_amica(
        X=data,
        max_iter=200,
        w_init=initial_weights,
        sbeta_init=initial_scales,
        mu_init=initial_locations,
        )

# %%
A_fortran = fortran_results['A']

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
fig1, ax1 = plot_topomaps(results["A"], output="python")


# %%
fig2, ax2 = plot_topomaps(A_fortran, output="fortran")