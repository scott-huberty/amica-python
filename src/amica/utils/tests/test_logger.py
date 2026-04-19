import logging

import pytest

from amica.utils import _logging, logger


@pytest.mark.parametrize("verbose,expected", [
    (None, "INFO"),
    (True, "INFO"),
    (False, "WARNING"),
    ("debug", "DEBUG"),
    (20, "INFO"),
    (logging.DEBUG, "DEBUG"),
])
def test_set_log_level_coercion(verbose, expected, capsys):
    """Ensure bools and None are coerced correctly."""
    _logging.set_log_level(verbose)
    logger.log(expected, "message")
    out = capsys.readouterr().out
    assert expected in out
    if verbose in (None, True):
        assert "INFO" in out
    elif verbose is False:
        assert "WARNING" in out
    elif verbose == 20:
        assert "INFO" in out
    elif verbose == logging.DEBUG:
        assert "DEBUG" in out


def test_log_preserves_literal_brackets(capsys):
    """Bracketed data should stay literal unless helper styling enables markup."""
    _logging.set_log_level("INFO")
    _logging.log("minimum eigenvalues: [1.0 2.0 3.0]")
    out = capsys.readouterr().out
    assert "minimum eigenvalues: [1.0 2.0 3.0]" in out
