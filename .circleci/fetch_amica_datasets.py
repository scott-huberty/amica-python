#!/usr/bin/env python3
"""Fetch AMICA datasets for CI and print the cache tree for debugging."""

import argparse
from pathlib import Path

from amica.datasets import data_path
from amica.utils import fetch_planck_temperature_maps
from amica.utils.fetch import PLANCK_MAP_FILENAMES

PLANCK_FREQUENCIES_GHZ = (30, 44, 70, 100, 143, 217)


def print_tree(directory: Path, prefix: str = "") -> None:
    """Recursively print a visual directory tree."""
    path = Path(directory)
    if not path.exists():
        print(f"{prefix}[missing] {path}")
        return

    items = sorted(
        path.iterdir(),
        key=lambda item: (not item.is_dir(), item.name.lower()),
    )

    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = "\\-- " if is_last else "|-- "
        print(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            child_prefix = prefix + ("    " if is_last else "|   ")
            print_tree(item, child_prefix)


def print_tree_section(directory: Path) -> None:
    """Fence the directory tree in CI logs so it is easy to spot."""
    print("=" * 80)
    print(f"AMICA dataset cache tree: {directory}")
    print("=" * 80)
    print_tree(directory)
    print("=" * 80)


def default_required_paths(cache_dir: Path) -> list[Path]:
    """Return required files for the default non-Planck dataset cache."""
    return [
        cache_dir / "eeglab_sample_data" / "eeglab_data.set",
        cache_dir / "eeglab_sample_data" / "eeglab_data.fdt",
        cache_dir / "eeglab_sample_data" / "amicaout_test" / "W",
        cache_dir / "photos" / "example2_lena",
    ]


def fetch_default_datasets(cache_dir: Path) -> None:
    """Fetch the default AMICA datasets used by tests and most docs examples."""
    required_paths = default_required_paths(cache_dir)

    if all(path.exists() for path in required_paths):
        print(f"Using cached default AMICA datasets from {cache_dir}")
        return

    resolved = data_path()
    print(f"Fetched default AMICA datasets into {resolved}")


def fetch_planck_frequency(cache_dir: Path, frequency_ghz: int) -> None:
    """Fetch one Planck frequency map for docs, if it is not already cached."""
    map_filename = PLANCK_MAP_FILENAMES[frequency_ghz]
    map_path = cache_dir / "planck_pr3" / map_filename

    if map_path.exists():
        print(f"Using cached Planck {frequency_ghz} GHz map from {map_path}")
        return

    fetch_planck_temperature_maps((frequency_ghz,))
    print(f"Fetched Planck {frequency_ghz} GHz map into {map_path}")


def parse_args() -> argparse.Namespace:
    """Parse CI dataset fetch options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planck-frequency",
        type=int,
        choices=PLANCK_FREQUENCIES_GHZ,
        help="Fetch one Planck PR3 frequency map instead of default datasets.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch the requested CI dataset group."""
    args = parse_args()
    cache_dir = Path.home() / "amica_test_data"
    print_tree_section(cache_dir)

    if args.planck_frequency is None:
        fetch_default_datasets(cache_dir)
    else:
        fetch_planck_frequency(cache_dir, args.planck_frequency)

    print_tree_section(cache_dir)


if __name__ == "__main__":
    main()
