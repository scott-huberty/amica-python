#!/usr/bin/env python
# coding: utf-8

"""
Use AMICA in a Scikit-Learn Pipeline
====================================

We'll use AMICA as a preprocessing step in a scikit-learn pipeline to perform
digit classification on the MNIST dataset.
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from amica import AMICA

# %%
# Load & split dataset

# %%
# Download MNIST (70k samples, 28×28 flattened)
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Just take digits 0-3 to speed up computation
mask = np.isin(y, ["0", "1", "2", "3"])
X = X[mask].copy()
y = y[mask].copy().astype(int)

# Train/test split: 60k / 10k
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7.0, shuffle=True, random_state=0
)

# %%
# Build scikit-learn pipeline with AMICA
# ---------------------------------------
pipe = Pipeline([
    ("center", StandardScaler(with_std=False)),  # remove global brightness bias
    ("amica", AMICA(n_components=60, max_iter=200, tol=.0001, random_state=0)),
    ("scale_components", StandardScaler()),      # optional but helps LR
    ("logreg", LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )),
])

# %%
# Fit
# ---

# %%
pipe.fit(X_train, y_train)

# %%
# Evaluate
# --------
y_pred = pipe.predict(X_test)

print(classification_report(
    y_test, y_pred, target_names=[str(i) for i in range(4)]
))

print(f"Accuracy: {pipe.score(X_test, y_test):.4f}")

# %%
# Important features for the 0 digit
# ----------------------------------
# We can select the most important ICA features for the 0 class (with negative and positive weights) and display their associate ICA sources.
#

# %%
# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def imshow_row(images, titles=None, figsize=(20, 4), suptitle=None, cmap="gray"):
    fig, axes = plt.subplots(1, len(images), figsize=figsize, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.15, va="top")
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(28, 28), cmap=cmap)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    return fig


# %%
# -----------------------------------------------------------------------------
# Show sample digits of class 0
# -----------------------------------------------------------------------------
zeros = X[y == 0][:10]

imshow_row(
    zeros,
    suptitle="10 samples of digit '0'"
)
plt.show()

# %%
# -----------------------------------------------------------------------------
# Top positive / negative logistic weights
# -----------------------------------------------------------------------------
logreg = pipe.named_steps["logreg"]
amica = pipe.named_steps["amica"]

coef = logreg.coef_[0]
sorted_idx = np.argsort(coef)

top_pos = sorted_idx[-5:][::-1]
top_neg = sorted_idx[:5]

imshow_row(
    amica.components_[top_pos],
    titles=[f"Comp {i}" for i in top_pos],
    suptitle="Top 5 positive AMICA components for class 0"
)
plt.show()

# %%
imshow_row(
    amica.components_[top_neg],
    titles=[f"Comp {i}" for i in top_neg],
    suptitle="Top 5 negative AMICA components for class 0"
)
plt.show()
