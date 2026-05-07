import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scentree.config import explained_var
from scentree.dim_reduction.pca import BasePCA
from scentree.estimators.estimator_orchestrator import EstimatorController
from scentree.io.loader import ValueRange
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, TypedDict


class ScenarioFanData(TypedDict):
    """
    Data container representing a scenario fan in a stochastic simulation framework.

    A scenario fan is a collection of independently generated scenario realizations
    derived from a common initial state. Each scenario represents a possible outcome
    of the underlying stochastic process.

    Attributes:
        scenarios (List[NDArray[np.float64]]): Generated scenarios.
        predicted_values (List[NDArray[np.float64]]): Predicted values associated
            with each scenario.
        observed_values Optional[List[NDArray[np.float64]]]: Observed values
            corresponding to each scenario, if available.
    """

    scenarios: List[NDArray[np.float64]]
    predicted_values: List[NDArray[np.float64]]
    observed_values: Optional[List[NDArray[np.float64]]]


class StageManager(BaseModel):
    """
    Orchestrates the generation of a scenario fan from historical data and model estimates.

    The StageManager is responsible for transforming historical time series data into
    a stochastic representation of future uncertainty. It performs preprocessing,
    dimensionality reduction, residual modeling, scenario generation, and reconstruction
    back to the original feature space.
    """

    model_config = {"arbitrary_types_allowed": True}

    def get_scenarios(
        self,
        residuals: NDArray[np.float64],
        estimated_values: NDArray[np.float64],
        num_fans: int,
        num_scenarios: int,
        seed: Optional[int] = None,
    ) -> List[NDArray[np.float64]]:
        """
        Obtain the scenarios given all components that are needed.

        Args:
            residuals(NDArray[np.float64]): Matrix containing historical residuals.
            estimated_values (NDArray[np.float64]): Estimated values.
            num_fans (int): Number of fans to provide.
            num_scenarios (int): Number of scenarios to generate the fan.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Returns:
            List[NDArray[np.float64]]: List containing the scenarios.
        """
        scenarios = []
        # For each day, generate the scenarios
        for i_day in range(num_fans):
            estimated_value = np.reshape(estimated_values[i_day, :], (1, -1))
            vector_ones = np.ones((num_scenarios, 1))
            # Transform it to a matrix. Each row is repeated
            estimated_matrix = np.matmul(vector_ones, estimated_value)
            # Take randonmly a sample of residuals
            rng = np.random.default_rng(seed)
            idx_residuals = rng.choice(a=residuals.shape[0], size=num_scenarios, replace=True)
            current_residuals = residuals[idx_residuals, :]
            current_scenarios = estimated_matrix + current_residuals
            scenarios.append(current_scenarios)
        return scenarios

    def clip_matrix_values(self, X: NDArray, value_ranges: Optional[List[ValueRange]]) -> NDArray:
        """
        Clip the values of a given matrix.

        Args:
            X (NDArray): Matrix to be clipped.
            value_ranges (Optional[List[ValueRange]]): Range of values used to clip X.
        """
        X_clipped = X.copy()
        if value_ranges is not None:
            for i, vr in enumerate(value_ranges):
                if vr is not None:
                    X_clipped[:, i] = np.clip(X[:, i], vr[0], vr[1])
        return X_clipped

    def generate_scenario_fans(
        self,
        X: NDArray[np.float64],
        num_fans: int,
        num_scenarios: int,
        build_in_sample_fans: bool = True,
        value_ranges: Optional[List[ValueRange]] = None,
        seed: Optional[int] = None,
    ) -> ScenarioFanData:
        """
        Orchestrates the scenario fan generation process from historical data.

        Args:
            X (NDArray[np.float64]): Historical data.
            num_fans (int):  Number of fans to generate.
            num_scenarios (int): Number of scenarios to generate.
            build_in_sample_fans (bool): Whether to build in-sample fans or
                out-sample fans. Default to True, meaning that in sample fans are built.
            value_ranges (Optional[List[ValueRange]]): Optional lower and upper bounds
                for each variable in `X`.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Raises:
            ValueError:
                - If the length of `value_ranges` does not match the number
                  of columns in `X`.
                - If a value range is invalid, i.e. the lower bound is
                  greater than the upper bound.

        Returns:
            ScenarioFanData:
                A container with:
                    - scenarios: generated scenarios
                    - predicted_values: model predictions
                    - observed_values: optional observed values
        """
        if value_ranges is not None:
            if len(value_ranges) != X.shape[1]:
                raise ValueError(
                    "The length of `value_ranges` must be equal to the number of columns of `X`"
                )
            for i, rv in enumerate(value_ranges):
                if rv is not None and rv[0] > rv[1]:
                    raise ValueError(
                        f"The first value of `value_ranges` must be greater than the second value in position {i}"
                    )
        # Perform normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform dimensionality reduction
        dim_reduction = BasePCA()
        X_reduced = dim_reduction.fit_auto_components(X_scaled, threshold=explained_var)

        # Find the best estimator
        estimator_controller = EstimatorController()
        estimator_controller.fit(X=X_reduced)

        # Estimate the residuals
        residuals = estimator_controller.estimate_residuals(X_reduced)

        # Get estimated values
        if build_in_sample_fans:
            estimated_values = estimator_controller.in_sample_estimation(num_fans)
            observed_values = [row for row in X[-num_fans:, :]]
        else:
            estimated_values = estimator_controller.out_sample_estimation(num_fans)
            observed_values = None

        # Create scenarios for all num_fans
        scenarios = self.get_scenarios(
            residuals,
            estimated_values,
            num_fans,
            num_scenarios,
            seed,
        )
        # For each fan, recover the data in the high dimensional space
        scenarios_high = []
        estimated_values_high = dim_reduction.inverse_transform(estimated_values)
        estimated_original = scaler.inverse_transform(estimated_values_high)
        for current_scenarios in scenarios:
            sc_high = dim_reduction.inverse_transform(current_scenarios)
            sc_original = scaler.inverse_transform(sc_high)
            sc_original_clipped = self.clip_matrix_values(sc_original, value_ranges)
            scenarios_high.append(sc_original_clipped)
        results: ScenarioFanData = {
            "scenarios": scenarios_high,
            "predicted_values": [row for row in estimated_original],
            "observed_values": observed_values,
        }
        return results
