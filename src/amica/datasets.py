"""Functions for accessing AMICA datasets."""
from pathlib import Path

from amica.utils import fetch_datasets


def data_path() -> Path:
    """Get the path to the directory containing AMICA's test data.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the EEGLAB test data and fortran outputs.
    """
    return fetch_datasets()
