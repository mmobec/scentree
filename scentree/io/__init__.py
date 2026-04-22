from typing import List, Optional, TypedDict


class MapColumnsStages(TypedDict):
    """
    Mapping between a dataset and the ordering of its columns.

    Attributes:
        dataset (str): Name of the dataset.
        columns (List[int]): List of column indices defining their order.
        stages (List[int]): List of column stages.
    """

    dataset: str
    columns: List[int]
    stages: List[int]


# Optional list of dataset-to-column mappings
MapColsNamesStages = Optional[List[MapColumnsStages]]
