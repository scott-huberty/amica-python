"""Verbosity helpers for AMICA public APIs."""


def _validate_verbose(verbose: int | bool | None) -> int:
    """Validate the AMICA verbosity contract and coerce legacy bool/None values."""
    if verbose is None:
        return 1
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    if not isinstance(verbose, int):
        raise TypeError("verbose must be an int in {0, 1, 2}")
    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be one of {0, 1, 2}")
    return verbose
