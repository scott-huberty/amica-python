"""Utilities for downloading and caching test datasets for AMICA Python comparisons."""

from pathlib import Path

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
    fetch_fortran_outputs()
    fetch_photos()
    return CACHE_DIR


def fetch_test_data() -> Path:  # pragma: no cover
    """
    Download the EEGLAB sample dataset (.set + .fdt).

    Returns
    -------
    dict
        Keys: "set", "fdt"
        Values: pathlib.Path objects to the cached files
    """
    import pooch

    sample_dir = CACHE_DIR / "eeglab_sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for _, (fname, known_hash) in EEGLAB_FILES.items():
        url = f"{EEGLAB_BASE}{fname}"
        _ = pooch.retrieve(
            url=url,
            known_hash=known_hash,
            path=sample_dir,
            fname=fname,
            progressbar=True,
        )
    return sample_dir


# -------------------------------
# Fortran golden outputs
# -------------------------------
version = "v0.6.0"
FORTRAN_URL = (
    "https://github.com/scott-huberty/amica/"
    f"releases/download/{version}/test_output.tar.gz"
)
FORTRAN_HASH = "sha256:46ec71a0f66565a43480825f85611ea0126e1aa05eaa52ceaf7b628631d753c1"


def fetch_fortran_outputs() -> Path:
    """
    Download and extract golden outputs from the Fortran implementation.

    Returns
    -------
    list of pathlib.Path
        Paths to the extracted files inside the tarball.
    """
    import pooch

    unpack = pooch.Untar(extract_dir=".")
    outputs_dir = pooch.retrieve(
        url=FORTRAN_URL,
        known_hash=FORTRAN_HASH,
        path=CACHE_DIR,
        processor=unpack,
        progressbar=True,
    )
    return Path(outputs_dir[0]).parent  # return the directory containing the files


# -------------------------------
# Photos dataset
# -------------------------------
COCKTAIL_BASE = "https://github.com/marcromani/cocktail/raw/refs/heads/master/examples/data/" # noqa E501
PHOTOS_FILES = {
    "example2_baboon": ("example2_baboon", "md5:b38c092ca8fda06e29182926866a1950"),
    "example2_cameraman": ("example2_cameraman", "md5:cb541d1814d3e27a4133d31653bbb01a"), # noqa E501
    "example2_lena": ("example2_lena", "md5:ce2f1b9f96561a409a5afc10654ab744"),
    "example2_mona": ("example2_mona", "md5:01f7b4cc8cade346843caac334d793d7"),
    "example2_texture": ("example2_texture", "md5:65c169b03e55686a66f6fb6471ca7d60"),
}

def fetch_photos() -> Path:
    """
    Download the photos dataset used in cocktail party example.

    Returns
    -------
    pathlib.Path
        Path to the directory containing all cached photo files.
    """
    import pooch

    photos_dir = CACHE_DIR / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    for _, (fname, known_hash) in PHOTOS_FILES.items():
        url = f"{COCKTAIL_BASE}{fname}"
        _ = pooch.retrieve(
            url=url,
            known_hash=known_hash,
            path=photos_dir,
            fname=fname,
            progressbar=True,
        )
    return photos_dir
