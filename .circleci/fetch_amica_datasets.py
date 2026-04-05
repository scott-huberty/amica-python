from pathlib import Path

from amica.datasets import data_path
from amica.utils import fetch_planck_temperature_maps
from amica.utils.fetch import PLANCK_MAP_FILENAMES

PLANCK_FREQUENCIES_GHZ = (30, 44, 70, 100, 143, 217)


def main() -> None:
    cache_dir = Path.home() / "amica_test_data"
    required_paths = [
        cache_dir / "eeglab_sample_data" / "eeglab_data.set",
        cache_dir / "eeglab_sample_data" / "eeglab_data.fdt",
        cache_dir / "photos" / "example2_lena",
    ]
    required_paths.extend(
        cache_dir / "planck_pr3" / PLANCK_MAP_FILENAMES[frequency_ghz]
        for frequency_ghz in PLANCK_FREQUENCIES_GHZ
    )

    if all(path.exists() for path in required_paths):
        print(f"Using cached AMICA datasets from {cache_dir}")
        return

    resolved = data_path()
    fetch_planck_temperature_maps(PLANCK_FREQUENCIES_GHZ)
    print(f"Fetched AMICA datasets into {resolved}")


if __name__ == "__main__":
    main()
