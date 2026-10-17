import numpy as np
from importlib.resources import as_file, files


def load_flowergirl() -> np.ndarray:
    """Load the Flower Girl dataset.

    The dataset is loaded from the package's data directory and split into
    two parts: all rows except the last one, and the last row.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The dataset excluding the last row.
            - The last row of the dataset.
    """
    data_path = files("scentree").joinpath("data/flowergirl.csv")

    with as_file(data_path) as path:
        flowergirl_data = np.loadtxt(path, delimiter=",")

    return flowergirl_data
