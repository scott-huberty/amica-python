from pathlib import Path

from amica.datasets import data_path


def main() -> None:
    cache_dir = Path.home() / "amica_test_data"
    required_paths = (
        cache_dir / "eeglab_sample_data" / "eeglab_data.set",
        cache_dir / "eeglab_sample_data" / "eeglab_data.fdt",
        cache_dir / "photos" / "example2_lena",
    )

    if all(path.exists() for path in required_paths):
        print(f"Using cached AMICA datasets from {cache_dir}")
        return

    resolved = data_path()
    print(f"Fetched AMICA datasets into {resolved}")


if __name__ == "__main__":
    main()
