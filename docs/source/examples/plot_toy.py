"""
Run ICA On Toy Data
===================
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import FastICA

import amica

# %%
# Generate Data and Load AMICA Results for Comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# %%
data_dir = amica.datasets.data_path() / "toy_2" / "amicaout_toy_2"

# %%
x = amica.utils.generate_toy_data(n_samples=10_000, noise_factor=.05)

# %%
# Run AMICA and FastICA for comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
fi = FastICA()
z = fi.fit_transform(x)

# %%
results = amica.fit_amica(
    x.copy(), centering=False, whiten=False, random_state=42,
)

# %%
# apply the learned unmixing matrix to the data
W = results["W"][:, :, 0]
y = np.dot(x, W)

# %%
fortran_W = np.fromfile(data_dir / "W", dtype=np.float64).reshape((2, 2), order="F")
y2 = np.dot(x, fortran_W)


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

