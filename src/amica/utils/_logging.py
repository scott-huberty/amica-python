import sys

from loguru import logger

# Remove default loguru handler so we control formatting
logger.remove()
FORMAT = "<level>{level: <8}</level> | {message} - <cyan>{name}</cyan>:<cyan>{function}</cyan>"  # noqa E501

# Add our own simple handler
logger.add(
    sys.stdout,
    level="INFO",  # default
    colorize=True,
    format=FORMAT,
)

def set_log_level(verbose: str | int) -> None:
    """Set global log level for amica.

    Parameters
    ----------
    verbose : str or int or bool or None, default=None
        Control verbosity of the logging output. If a str, it can be either
        ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.
        Note that these are for convenience and are equivalent to passing in
        ``logging.DEBUG``, etc. For ``bool``, ``True`` is the same as ``"INFO"``,
        ``False`` is the same as ``"WARNING"``. If ``None``, defaults to ``"INFO"``.
    """
    if verbose is None:
        verbose = "INFO"
    elif isinstance(verbose, bool):
        verbose = "INFO" if verbose else "WARNING"
    elif isinstance(verbose, str):
        verbose = verbose.upper()
    logger.remove()
    logger.add(
        sys.stdout,
        level=verbose,
        colorize=True,
        format=FORMAT,
    )
