
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd


@dataclass
class ScalingRules:
    """
    Configurable rules for numerical feature scaling.
    """

    strategies: dict[str, str] = field(
        default_factory=dict
    )

    feature_ranges: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )


@dataclass
class ScalingReport:
    """
    Contains the complete dataset scaling results.
    """

    original_columns: list[str] = field(
        default_factory=list
    )

    scaled_columns: list[str] = field(
        default_factory=list
    )

    strategies_applied: dict[str, str] = field(
        default_factory=dict
    )

    rows_processed: int = 0


@dataclass
class ScalingResult:
    """
    Contains the scaled dataset and scaling report.
    """

    dataframe: pd.DataFrame
    report: ScalingReport

    scaling_parameters: dict[str, dict] = field(
        default_factory=dict
    )


def scale_dataset(
    dataframe: pd.DataFrame,
    rules: ScalingRules | None = None,
) -> ScalingResult:
    """
    Scale numerical DataFrame columns using configurable strategies.

    Supported scaling strategies:
    - min_max
    - standard

    Min-Max Scaling transforms values into a configurable range.
    Default range is (0, 1).

    Standard Scaling transforms values using mean and
    population standard deviation (ddof=0).

    The original DataFrame is never modified.
    """

    # Validate input DataFrame.
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if not dataframe.columns.is_unique:
        raise ValueError(
            "DataFrame must have unique column names."
        )

    # Apply default configuration.
    if rules is None:
        rules = ScalingRules()

    if not isinstance(rules, ScalingRules):
        raise TypeError(
            "Expected ScalingRules configuration."
        )

    valid_strategies = {
        "min_max",
        "standard",
    }

    # Validate scaling configurations.
    for column, strategy in rules.strategies.items():

        if column not in dataframe.columns:
            raise ValueError(
                f"Scaling rule references unknown column '{column}'."
            )

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported scaling strategy '{strategy}'."
            )

        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise TypeError(
                f"Scaling requires a numeric column: '{column}'."
            )

        if dataframe[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values. "
                "Clean missing values before scaling."
            )

        if not np.isfinite(
            dataframe[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )

    # Validate feature ranges.
    for column, feature_range in rules.feature_ranges.items():

        if column not in rules.strategies:
            raise ValueError(
                f"Feature range provided for '{column}' "
                "without a scaling strategy."
            )

        if rules.strategies[column] != "min_max":
            raise ValueError(
                f"Feature ranges can only be used "
                f"with min_max scaling: '{column}'."
            )

        if (
            not isinstance(feature_range, tuple)
            or len(feature_range) != 2
        ):
            raise ValueError(
                f"Feature range for '{column}' "
                "must be a tuple of two values."
            )

        lower, upper = feature_range

        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, Real)
            or not isinstance(upper, Real)
        ):
            raise ValueError(
                f"Feature range values for '{column}' "
                "must be real numbers."
            )

        if not np.isfinite([lower, upper]).all():
            raise ValueError(
                f"Feature range values for '{column}' "
                "must be finite."
            )

        if lower >= upper:
            raise ValueError(
                f"Feature range minimum must be less "
                f"than maximum for '{column}'."
            )

    # Capture original dataset information.
    original_columns = list(dataframe.columns)

    # Create an independent copy.
    scaled = dataframe.copy(deep=True)

    # Initialize reports and parameters.
    scaled_columns = []
    strategies_applied = {}
    scaling_parameters = {}

    # Apply scaling strategies.
    for column, strategy in rules.strategies.items():

        values = scaled[column]

        # Min-Max Scaling.
        if strategy == "min_max":

            minimum = float(values.min())
            maximum = float(values.max())

            lower, upper = rules.feature_ranges.get(
                column, (0.0, 1.0)
            )

            if minimum == maximum:
                raise ValueError(
                    f"Cannot apply min_max scaling to "
                    f"constant column '{column}'."
                )

            scaled[column] = (
                (values - minimum)
                / (maximum - minimum)
            ) * (upper - lower) + lower

            scaling_parameters[column] = {
                "strategy": strategy,
                "min": minimum,
                "max": maximum,
                "feature_range": (lower, upper),
            }

        # Standard Scaling.
        elif strategy == "standard":

            mean = float(values.mean())
            std = float(values.std(ddof=0))

            if std == 0:
                raise ValueError(
                    f"Cannot apply standard scaling to "
                    f"constant column '{column}'."
                )

            scaled[column] = (
                values - mean
            ) / std

            scaling_parameters[column] = {
                "strategy": strategy,
                "mean": mean,
                "std": std,
            }

        scaled_columns.append(column)
        strategies_applied[column] = strategy

    # Generate scaling report.
    report = ScalingReport(
        original_columns=original_columns,
        scaled_columns=scaled_columns,
        strategies_applied=strategies_applied,
        rows_processed=len(scaled),
    )

    return ScalingResult(
        dataframe=scaled,
        report=report,
        scaling_parameters=scaling_parameters,
    )
