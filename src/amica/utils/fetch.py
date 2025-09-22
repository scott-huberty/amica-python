"""
Utilities for downloading and caching test datasets
for AMICA Python comparisons.
"""

from pathlib import Path
import pooch


# Cache directory for all test data
CACHE_DIR = Path.home() / "amica_test_data"


# -------------------------------
# EEGLAB test data
# -------------------------------
EEGLAB_BASE = "https://github.com/sccn/eeglab/raw/develop/sample_data/"
EEGLAB_FILES = {
    "fdt": ("eeglab_data.fdt", "md5:a135e79e2acc93670746b2b6f44570a7"),
    "set": ("eeglab_data.set", "md5:cf2d4549e48b8fd82cff776d13adb46c"),
}


def fetch_datasets() -> Path:
    """
    Download all test datasets.

    Returns
    -------
    pathlib.Path
        Path to the directory containing all cached test datasets.
    """
    fetch_test_data()
    fetch_fortran_outputs()
    return CACHE_DIR


def fetch_test_data() -> Path:
    """
    Download the EEGLAB sample dataset (.set + .fdt).

    Returns
    -------
    dict
        Keys: "set", "fdt"
        Values: pathlib.Path objects to the cached files
    """
    for _, (fname, known_hash) in EEGLAB_FILES.items():
        url = f"{EEGLAB_BASE}{fname}"
        fpath = pooch.retrieve(
            url=url,
            known_hash=known_hash,
            path=CACHE_DIR,
            fname=fname,
            progressbar=True,
        )
    return Path(fpath).parent  # return the directory containing the files


# -------------------------------
# Fortran golden outputs
# -------------------------------
version = "v0.1.0"
FORTRAN_URL = (
    "https://github.com/scott-huberty/amica/"
    f"releases/download/{version}/amicaout_test.tar.gz"
)
FORTRAN_HASH = "md5:9b0b4beb1a669dd4dbcd12d3b398376e"


def fetch_fortran_outputs() -> Path:
    """
    Download and extract golden outputs from the Fortran implementation.

    Returns
    -------
    list of pathlib.Path
        Paths to the extracted files inside the tarball.
    """
    unpack = pooch.Untar(extract_dir="amicaout_test")
    outputs_dir = pooch.retrieve(
        url=FORTRAN_URL,
        known_hash=FORTRAN_HASH,
        path=CACHE_DIR,
        processor=unpack,
        progressbar=True,
    )
    return Path(outputs_dir[0]).parent  # return the directory containing the files
