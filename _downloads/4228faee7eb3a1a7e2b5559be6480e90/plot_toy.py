"""
Run ICA On Toy Data
===================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import FastICA

import amica
from amica import AMICA

# %%
# Generate Data and Load AMICA Results for Comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# %%
data_dir = amica.datasets.data_path() / "toy_2" / "amicaout_toy_2"

# %%
x = amica.utils.generate_toy_data(n_samples=10_000, noise_factor=.05, seed=42)

# %%
# Run AMICA and FastICA for comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
fi = FastICA()
z = fi.fit_transform(x)

# %%
transformer = AMICA(mean_center=False, whiten="variance", random_state=42,)
transformer.fit(x.copy())

# %%
# apply the learned unmixing matrix to the data
y = transformer.transform(x)

# %%
fortran_results = amica.utils.load_fortran_results(
    data_dir, n_components=2, n_mixtures=3
    )
W_f = fortran_results["W"][:, :, 0]
y2 = x @ (W_f @ fortran_results["S"]).T


# %%
# Plot Results
# ^^^^^^^^^^^^

# %%
fig, ax = plt.subplots(4, 1, sharex=True)
for i, l, v in zip([1, 2, 3, 4], ['Sources', 'AMICA', 'AMICA-Python', 'FastICA'], [x, y2, y, z]):
    ax = plt.subplot(4, 1, i)
    ax.plot(v[:1000])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(l)
    ax.set_ylim([-1.1 * np.max(v), 1.1 * np.max(v)])

