from scentree.estimators.scikit_base import SklearnEstimator
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from typing import Optional, Type


class RidgeEstimator(SklearnEstimator):
    """Wrapper for the scikit-learn `Ridge` regression estimator.

    Args:
    min_variables (int): Minimum number of variables required to run the estimator.
    estimator_class (Optional[Type[BaseEstimator]]): Estimator class to wrap.
        Defaults to `sklearn.linear_model.Ridge`.
    """

    min_variables = 1

    def __init__(self, estimator_class: Optional[Type[BaseEstimator]] = None):
        if estimator_class is None:
            estimator_class = Ridge
        super().__init__(estimator_class=estimator_class)
