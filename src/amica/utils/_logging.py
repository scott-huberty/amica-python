import logging
import sys
from typing import Any

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.logging import RichHandler
from rich.theme import Theme

ACCENT_COLOR = "#9A5CD0"
GRASS = "#46A758"
YELLOW = "#FFE629"
ORANGE = "#F76B15"
SAND = "#62605B"
FORMAT = "%(message)s"
LOGGER_NAME = "amica"


class AmicaNumberHighlighter(RegexHighlighter):
    """Highlight numeric values with the AMICA accent color."""

    base_style = "amica."
    highlights = [
        r"(?P<number>-?\d+(?:\.\d+)?)",
    ]


def _coerce_level(verbose: str | int | bool | None) -> int:
    """Normalize supported verbosity inputs to a stdlib logging level."""
    if verbose is None:
        return logging.INFO
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    if isinstance(verbose, str):
        return logging._nameToLevel.get(verbose.upper(), logging.INFO)
    if isinstance(verbose, int):
        return verbose
    return logging.INFO


def _configure_logger(level: str | int | bool | None = logging.INFO) -> logging.Logger:
    """Create or update the shared amica logger."""
    configured_level = _coerce_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(configured_level)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = RichHandler(
        console=Console(file=sys.stdout, theme=None),
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        highlighter=AmicaNumberHighlighter(),
        log_time_format=None,
    )
    handler.setLevel(configured_level)
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.console.push_theme(
        Theme(
            {
                "amica.number": SAND,
                "logging.level.info": ACCENT_COLOR,
            }
        )
    )
    logger.addHandler(handler)
    return logger


def _style_markup(
        msg: str, *, color: str | None = None, weight: str | None = None
        ) -> tuple[str, bool]:
    """Return a Rich-markup message plus whether markup should be enabled."""
    styles: list[str] = []
    if weight == "bold":
        styles.append("bold")
    if color:
        styles.append(color)
    if not styles:
        return msg, False

    style = " ".join(styles)
    return f"[{style}]{msg}[/{style}]", True


class AmicaLogger:
    """Thin compatibility wrapper around the package logger."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._logger, name)
        if not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("stacklevel", 2)
            return attr(*args, **kwargs)

        return wrapped

    def log(self, level: str | int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Support both numeric and string log levels."""
        kwargs.setdefault("stacklevel", 2)
        self._logger.log(_coerce_level(level), msg, *args, **kwargs)


_base_logger = _configure_logger()
logger = AmicaLogger(_base_logger)


def get_logger() -> AmicaLogger:
    return logger


def set_log_level(verbose: str | int | bool | None) -> None:
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
    _configure_logger(verbose)


def log(
    msg: str,
    level: str = "info",
    color: str | None = None,
    weight: str | None = None,
) -> None:
    """Wrap the package logger with optional Rich markup."""
    msg, use_markup = _style_markup(msg, color=color, weight=weight)
    get_logger().log(level, msg, extra={"markup": use_markup})


def _emit_status(progress, msg: str, *, level: str = "info", color: str | None = None,
                 weight: str | None = None) -> None:
    """Emit important messages even when progress-bar mode suppresses normal logs."""
    if progress is None:
        log(msg, level=level, color=color, weight=weight)
        return

    msg, use_markup = _style_markup(msg, color=color, weight=weight)
    progress.console.print(msg, markup=use_markup)
