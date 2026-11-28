
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd


@dataclass
class ScalingRules:
    """Configurable rules for numerical feature scaling."""

    strategies: dict[str, str] = field(
        default_factory=dict
    )

    feature_ranges: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )


@dataclass
class ScalingReport:
    """Contains the complete dataset scaling results."""

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
    """Contains the scaled dataset and scaling report."""

    dataframe: pd.DataFrame
    report: ScalingReport

    scaling_parameters: dict[str, dict] = field(
        default_factory=dict
    )


class DataScaler:
    """
    Reusable numerical feature scaling engine.

    Supported strategies:
    - min_max
    - standard

    Lifecycle:
    - fit()
    - transform()
    - fit_transform()
    - get_state()

    Original DataFrames are never modified.
    """

    VALID_STRATEGIES = {
        "min_max",
        "standard",
    }

    def __init__(
        self,
        rules: ScalingRules | None = None,
    ):

        if rules is None:
            rules = ScalingRules()

        if not isinstance(rules, ScalingRules):
            raise TypeError(
                "Expected ScalingRules configuration."
            )

        self.rules = rules

        self._scaling_parameters = {}
        self._fitted = False

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    def _validate_dataframe(self, dataframe):

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Expected a Pandas DataFrame."
            )

        if not dataframe.columns.is_unique:
            raise ValueError(
                "DataFrame must have unique column names."
            )

    def _validate_configuration(self, dataframe):

        self._validate_dataframe(dataframe)

        for column, strategy in self.rules.strategies.items():

            if column not in dataframe.columns:
                raise ValueError(
                    f"Scaling rule references unknown "
                    f"column '{column}'."
                )

            if strategy not in self.VALID_STRATEGIES:
                raise ValueError(
                    f"Unsupported scaling strategy '{strategy}'."
                )

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise TypeError(
                    f"Scaling requires a numeric column: "
                    f"'{column}'."
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

        for column, feature_range in (
            self.rules.feature_ranges.items()
        ):

            if column not in self.rules.strategies:
                raise ValueError(
                    f"Feature range provided for '{column}' "
                    "without a scaling strategy."
                )

            if self.rules.strategies[column] != "min_max":
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

    # --------------------------------------------------
    # FIT: LEARN SCALING PARAMETERS
    # --------------------------------------------------

    def fit(self, dataframe: pd.DataFrame):
        """
        Learn scaling parameters from training data.
        """

        self._validate_configuration(dataframe)

        learned_parameters = {}

        for column, strategy in self.rules.strategies.items():

            values = dataframe[column]

            if strategy == "min_max":

                minimum = float(values.min())
                maximum = float(values.max())

                if minimum == maximum:
                    raise ValueError(
                        f"Cannot apply min_max scaling to "
                        f"constant column '{column}'."
                    )

                lower, upper = self.rules.feature_ranges.get(
                    column, (0.0, 1.0)
                )

                learned_parameters[column] = {
                    "strategy": strategy,
                    "min": minimum,
                    "max": maximum,
                    "feature_range": (lower, upper),
                }

            elif strategy == "standard":

                mean = float(values.mean())
                std = float(values.std(ddof=0))

                if std == 0:
                    raise ValueError(
                        f"Cannot apply standard scaling to "
                        f"constant column '{column}'."
                    )

                learned_parameters[column] = {
                    "strategy": strategy,
                    "mean": mean,
                    "std": std,
                }

        self._scaling_parameters = learned_parameters
        self._fitted = True

        return self

    # --------------------------------------------------
    # TRANSFORM: APPLY LEARNED PARAMETERS
    # --------------------------------------------------

    def transform(self, dataframe: pd.DataFrame):
        """
        Scale data using previously learned parameters.
        """

        if not self._fitted:
            raise RuntimeError(
                "DataScaler must be fitted before transform()."
            )

        self._validate_dataframe(dataframe)

        for column in self.rules.strategies:

            if column not in dataframe.columns:
                raise ValueError(
                    f"Required scaling column '{column}' "
                    "is missing from the DataFrame."
                )

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise TypeError(
                    f"Scaling requires a numeric column: "
                    f"'{column}'."
                )

            if dataframe[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values."
                )

            if not np.isfinite(
                dataframe[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Column '{column}' contains non-finite values."
                )

        scaled = dataframe.copy(deep=True)

        original_columns = list(dataframe.columns)

        scaled_columns = []
        strategies_applied = {}

        for column, strategy in self.rules.strategies.items():

            values = scaled[column]
            parameters = self._scaling_parameters[column]

            if strategy == "min_max":

                minimum = parameters["min"]
                maximum = parameters["max"]

                lower, upper = parameters["feature_range"]

                scaled[column] = (
                    (values - minimum)
                    / (maximum - minimum)
                ) * (upper - lower) + lower

            elif strategy == "standard":

                mean = parameters["mean"]
                std = parameters["std"]

                scaled[column] = (
                    values - mean
                ) / std

            scaled_columns.append(column)
            strategies_applied[column] = strategy

        report = ScalingReport(
            original_columns=original_columns,
            scaled_columns=scaled_columns,
            strategies_applied=strategies_applied,
            rows_processed=len(scaled),
        )

        return ScalingResult(
            dataframe=scaled,
            report=report,
            scaling_parameters=deepcopy(
                self._scaling_parameters
            ),
        )

    # --------------------------------------------------
    # FIT_TRANSFORM
    # --------------------------------------------------

    def fit_transform(self, dataframe: pd.DataFrame):
        """
        Learn parameters and transform training data.
        """

        self.fit(dataframe)

        return self.transform(dataframe)

    # --------------------------------------------------
    # STATE RETRIEVAL
    # --------------------------------------------------

    def get_state(self):
        """
        Return the current learned scaling state.
        """

        return {
            "fitted": self._fitted,
            "strategies": deepcopy(
                self.rules.strategies
            ),
            "scaling_parameters": deepcopy(
                self._scaling_parameters
            ),
        }


# ------------------------------------------------------
# BACKWARD-COMPATIBLE FUNCTION
# ------------------------------------------------------

def scale_dataset(
    dataframe: pd.DataFrame,
    rules: ScalingRules | None = None,
) -> ScalingResult:
    """
    Backward-compatible standalone scaling function.
    """

    scaler = DataScaler(rules)

    return scaler.fit_transform(dataframe)

