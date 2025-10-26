
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from amica import AMICA


@pytest.mark.sklearn_api
@parametrize_with_checks([AMICA(random_state=0)])
def test_check_estimator(estimator, check):
    """Test Scikit-Learn API Compatibility."""
    check(estimator)
