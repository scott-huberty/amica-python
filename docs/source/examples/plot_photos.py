"""
Blind source separation (BSS) on blended photographs
====================================================

We blend 5 grayscale photographs using a random mixing matrix
and attempt to recover them using AMICA and FastICA.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import FastICA

from amica import AMICA, datasets


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_grayscale_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size)
    return np.asarray(img, dtype=float) / 255.0


def plot_images(images: list[np.ndarray], title: str | None = None) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    if title:
        fig.suptitle(title, fontsize=14)

    for ax, img in zip(axes, images):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Load source images
# ---------------------------------------------------------------------
photos_dir = datasets.data_path() / "photos"
filenames = [
    "example2_baboon",
    "example2_cameraman",
    "example2_lena",
    "example2_mona",
    "example2_texture",
]

sources = [load_grayscale_image(photos_dir / fn) for fn in filenames]

# Common shape (ensure consistent size)
height, width = sources[0].shape

# Stack flattened as rows -> shape (n_sources, n_pixels)
S = np.vstack([src.ravel() for src in sources])

# ---------------------------------------------------------------------
# Mix sources with random mixing matrix
# ---------------------------------------------------------------------
seed = 42
rng = np.random.default_rng(seed)
A = rng.random((5, 5))
X = A @ S  # shape (5, pixels)

mixed_images = [row.reshape(height, width) for row in X]
plot_images(mixed_images, title="Mixed Observations")


# ---------------------------------------------------------------------
# Recover with AMICA
# ---------------------------------------------------------------------
amica = AMICA(random_state=seed, tol=.0001) # increase tol to match FastICA tolerance
S_amica = amica.fit_transform(X.T).T
recovered_amica = [row.reshape(height, width) for row in S_amica]

plot_images(recovered_amica, title="Recovered with AMICA")


# ---------------------------------------------------------------------
# Recover with FastICA
# ---------------------------------------------------------------------
fastica = FastICA(n_components=5, random_state=seed)
S_fast = fastica.fit_transform(X.T).T
recovered_fastica = [row.reshape(height, width) for row in S_fast]

plot_images(recovered_fastica, title="Recovered with FastICA")
