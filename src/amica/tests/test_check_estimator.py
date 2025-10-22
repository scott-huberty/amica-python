from sklearn.utils.estimator_checks import parametrize_with_checks

from amica import AMICA


@parametrize_with_checks([AMICA()])
def test_check_estimator(estimator, check):
    """Test Scikit-Learn API Compatibility."""
    check(estimator)
