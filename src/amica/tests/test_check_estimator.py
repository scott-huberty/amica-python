import sys

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from amica import AMICA


@pytest.mark.skipif(sys.platform == "win32", reason="Numerical Failures on Windows")
@parametrize_with_checks([AMICA()])
def test_check_estimator(estimator, check):
    """Test Scikit-Learn API Compatibility."""
    check(estimator)
