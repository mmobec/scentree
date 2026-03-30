import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error
from typing import Union


def rmse(
    y_true: Union[NDArray[np.float64], list], y_pred: Union[NDArray[np.float64], list]
) -> float:
    """Computes the root mean squared error.

    Args:
        y_true (Union[NDArray[np.float64], list]): Real value.
        y_pred (Union[NDArray[np.float64], list]): Predicted value.

    Returns:
        float: Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))
