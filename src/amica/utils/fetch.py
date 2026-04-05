"""Utilities for downloading and caching datasets for AMICA Python."""

import os
import tempfile
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
    Download the default AMICA test datasets.

    Notes
    -----
    This intentionally excludes the optional Planck astronomy maps because they
    are large and should not be fetched automatically for every user.

    Returns
    -------
    pathlib.Path
        Path to the directory containing all cached test datasets.
    """
    fetch_test_data()
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


# -------------------------------
# Optional Planck PR3 astronomy maps
# -------------------------------
PLANCK_BASE_URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/"  # noqa: E501
PLANCK_MAP_FILENAMES = {
    30: "LFI_SkyMap_030-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    44: "LFI_SkyMap_044-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    70: "LFI_SkyMap_070-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    100: "HFI_SkyMap_100_2048_R3.01_full.fits",
    143: "HFI_SkyMap_143_2048_R3.01_full.fits",
    217: "HFI_SkyMap_217_2048_R3.01_full.fits",
}


# -------------------------------
# Optional MICA benchmark dataset
# -------------------------------
MICA_RELEASE_URL = "http://sccn.ucsd.edu/pub/mica_release.zip"


def _prepare_cache_dir(root: Path, dataset_name: str) -> Path:
    """Create a writable cache directory, falling back when needed."""
    cache_dir = root / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(cache_dir, os.W_OK):
        raise PermissionError(f"Cache directory is not writable: {cache_dir}")
    return cache_dir


def fetch_planck_temperature_map(filename: str) -> Path:  # pragma: no cover
    """Download one public Planck PR3 map and return the local path."""
    import pooch

    cache_root = Path(os.environ.get("AMICA_PLANCK_CACHE", CACHE_DIR))
    fallback_cache_root = Path(tempfile.gettempdir()) / "amica-python"
    url = f"{PLANCK_BASE_URL}{filename}"

    try:
        cache_dir = _prepare_cache_dir(cache_root, "planck_pr3")
    except PermissionError:
        cache_dir = _prepare_cache_dir(fallback_cache_root, "planck_pr3")

    try:
        return Path(
            pooch.retrieve(
                url=url,
                known_hash=None,
                path=cache_dir,
                fname=filename,
                progressbar=True,
            )
        )
    except Exception as exc:  # pragma: no cover - depends on network/data host
        raise RuntimeError(
            f"Could not download the Planck map '{filename}' from {url}."
        ) from exc


def fetch_planck_temperature_maps(
    frequencies_ghz: tuple[int, ...] | None = None,
) -> dict[int, Path]:  # pragma: no cover
    """Download selected Planck PR3 temperature maps.

    Parameters
    ----------
    frequencies_ghz : tuple of int | None
        Requested Planck channel frequencies in GHz. If ``None``, downloads all
        channels listed in ``PLANCK_MAP_FILENAMES``.

    Returns
    -------
    dict of int to pathlib.Path
        Mapping from channel frequency in GHz to the local cached file path.
    """
    if frequencies_ghz is None:
        frequencies_ghz = tuple(PLANCK_MAP_FILENAMES)

    missing = sorted(set(frequencies_ghz) - set(PLANCK_MAP_FILENAMES))
    if missing:
        raise ValueError(
            f"Unsupported Planck frequencies requested: {missing}. "
            f"Available channels are {sorted(PLANCK_MAP_FILENAMES)}."
        )

    return {
        frequency_ghz: fetch_planck_temperature_map(PLANCK_MAP_FILENAMES[frequency_ghz])
        for frequency_ghz in frequencies_ghz
    }


def fetch_mica_release(output_dir: Path | None = None) -> Path:  # pragma: no cover
    """Download and extract the optional EEGLAB MICA benchmark dataset.

    This dataset is large, so it is intentionally excluded from
    :func:`fetch_datasets` and must be requested explicitly.

    Parameters
    ----------
    output_dir : pathlib.Path | None
        Directory where the extracted ``mica_release`` folder should live.
        Defaults to ``~/amica_test_data``.

    Returns
    -------
    pathlib.Path
        Path to the extracted ``mica_release`` directory.
    """
    import pooch

    cache_root = Path(output_dir).expanduser() if output_dir is not None else CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    zip_path = Path(
        pooch.retrieve(
            url=MICA_RELEASE_URL,
            known_hash=None,
            path=cache_root,
            fname="mica_release.zip",
            progressbar=True,
        )
    )

    release_dir = cache_root / "mica_release"
    if release_dir.exists():
        return release_dir

    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_root)

    if not release_dir.exists():
        raise RuntimeError(
            f"Expected extracted benchmark directory at {release_dir}, "
            f"but it was not created from {zip_path}."
        )

    return release_dir
