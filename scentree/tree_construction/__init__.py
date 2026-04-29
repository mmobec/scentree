import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, TypedDict, Union


class Node(TypedDict):
    """
    Represents a node in a scenario tree structure.

    Each node is uniquely identified by a `key` and may reference a parent
    node through `parent_key`. It contains metadata describing which scenarios
    it belongs to and an optional textual description.

    Attributes:
        key (Tuple[int, int]): Unique identifier of the node in the tree.
        scenario_ids (List[int]): List of scenario identifiers associated
            with this node.
        parent_key (Union[Tuple[int, int], None]): Identifier of the parent node
            in the tree. None if the node is a root node.
        description (str): Human-readable description of the node.
    """

    key: Tuple[Optional[int], int]
    scenario_ids: List[int]
    parent_key: Union[Tuple[Optional[int], Optional[int]], None]
    description: str


# A tree is a list of nodes.
Tree = List[Node]

# Mapping for trees.
NodeScenarioMap = Dict[Tuple[int, int], List[int]]


class ScenarioTree(TypedDict):
    """
    Represents a scenario tree structure used in stochastic programming.

    Attributes:
        tree (List[Node]):List of nodes defining the tree structure. Each node contains
            information about its position in the tree, its parent, and the
            scenarios associated with it.

        scenario_probabilities (NDArray[np.float64]): Array containing the probability
            of each scenario.

        scenario_tree_data (NDArray[np.float64]): Matrix containing the data associated
            with each scenario in the tree.
    """

    tree: List[Node]
    scenario_probabilities: NDArray[np.float64]
    scenario_tree_data: NDArray[np.float64]


# Collection of trees.
ScenarioTrees = List[ScenarioTree]
