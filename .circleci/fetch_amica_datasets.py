#!/usr/bin/env python3
"""Fetch AMICA datasets for CI and print the cache tree for debugging."""

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

    items = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))

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


def main() -> None:
    cache_dir = Path.home() / "amica_test_data"
    print_tree_section(cache_dir)
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
    print_tree_section(cache_dir)

if __name__ == "__main__":
    main()
