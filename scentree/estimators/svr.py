from scentree.estimators.scikit_base import SklearnEstimator
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from typing import Literal, Optional, Type


class MultiOutputSVR(MultiOutputRegressor):
    """Multi-output wrapper for the scikit-learn `SVR` regression estimator.

    Args:
        C (float): Regularization parameter of the underlying `SVR`.
            Defaults to `1.0`.
        epsilon (float): Defines the epsilon-tube within which no penalty is
            associated with errors in the training data. Defaults to `0.1`.
        gamma (float | Literal["scale", "auto"]): Kernel coefficient for the
            underlying `SVR`. Defaults to `"scale"`.
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: float | Literal["scale", "auto"] = "scale",
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        super().__init__(estimator=SVR(C=C, epsilon=epsilon, gamma=gamma))


class SVREstimator(SklearnEstimator):
    """Wrapper for a multi-output scikit-learn `SVR` regression estimator.

    Args:
        estimator_class (Optional[Type[BaseEstimator]]): Estimator class to wrap.
            Defaults to `MultiOutputSVR`.
    """

    def __init__(self, estimator_class: Optional[Type[BaseEstimator]] = None):
        if estimator_class is None:
            estimator_class = MultiOutputSVR
        super().__init__(estimator_class=estimator_class)
