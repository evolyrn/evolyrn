
from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class CleaningRules:
    """
    Configurable rules for dataset cleaning.
    """

    missing_value_strategies: dict[str, str] = field(
        default_factory=dict
    )

    fill_values: dict[str, object] = field(
        default_factory=dict
    )

    remove_duplicates: bool = True
    duplicate_keep: str = "first"


@dataclass
class CleaningReport:
    """
    Contains the complete dataset cleaning results.
    """

    original_row_count: int
    final_row_count: int

    missing_values_before: dict[str, int] = field(
        default_factory=dict
    )

    missing_values_after: dict[str, int] = field(
        default_factory=dict
    )

    duplicate_rows_before: int = 0
    duplicates_removed: int = 0

    rows_removed_missing: int = 0


@dataclass
class CleaningResult:
    """
    Contains the cleaned dataset and cleaning report.
    """

    dataframe: pd.DataFrame
    report: CleaningReport


class DataCleaner:
    """
    Reusable dataset cleaning engine.

    Supported missing-value strategies:
    - mean
    - median
    - mode
    - constant
    - drop_rows

    Workflow:
    - fit(): Learn cleaning parameters.
    - transform(): Apply learned parameters.
    - fit_transform(): Learn and apply parameters.

    The original DataFrame is never modified.
    """

    VALID_STRATEGIES = {
        "mean",
        "median",
        "mode",
        "constant",
        "drop_rows",
    }

    def __init__(
        self,
        rules: CleaningRules | None = None,
    ):
        if rules is None:
            rules = CleaningRules()

        if not isinstance(rules, CleaningRules):
            raise TypeError(
                "Expected CleaningRules configuration."
            )

        self.rules = rules

        self._is_fitted = False
        self._columns = []
        self._learned_parameters = {}

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate input DataFrame."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Expected a Pandas DataFrame."
            )

        if not dataframe.columns.is_unique:
            raise ValueError(
                "DataFrame must have unique column names."
            )

    def _validate_rules(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate cleaning configuration."""

        if self.rules.duplicate_keep not in (
            "first",
            "last",
        ):
            raise ValueError(
                "duplicate_keep must be either 'first' or 'last'."
            )

        for column, strategy in (
            self.rules.missing_value_strategies.items()
        ):

            if column not in dataframe.columns:
                raise ValueError(
                    f"Cleaning rule references unknown "
                    f"column '{column}'."
                )

            if strategy not in self.VALID_STRATEGIES:
                raise ValueError(
                    f"Unsupported cleaning strategy '{strategy}'."
                )

            if strategy in ("mean", "median"):

                if dataframe[column].isna().all():
                    raise ValueError(
                        f"Cannot calculate {strategy} for "
                        f"column '{column}' because all "
                        "values are missing."
                    )

                if not is_numeric_dtype(dataframe[column]):
                    raise TypeError(
                        f"Strategy '{strategy}' requires "
                        f"a numeric column: '{column}'."
                    )

            if (
                strategy == "constant"
                and column not in self.rules.fill_values
            ):
                raise ValueError(
                    f"Constant fill value not provided "
                    f"for '{column}'."
                )

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "DataCleaner":
        """
        Learn cleaning parameters from training data.
        """

        self._validate_dataframe(dataframe)
        self._validate_rules(dataframe)

        learned_parameters = {}

        for column, strategy in (
            self.rules.missing_value_strategies.items()
        ):

            if strategy == "mean":
                value = float(
                    dataframe[column].mean()
                )

            elif strategy == "median":
                value = float(
                    dataframe[column].median()
                )

            elif strategy == "mode":

                modes = dataframe[column].mode()

                if modes.empty:
                    raise ValueError(
                        f"Cannot calculate mode for "
                        f"column '{column}'."
                    )

                value = modes.iloc[0]

            elif strategy == "constant":
                value = self.rules.fill_values[column]

            else:
                value = None

            # Convert NumPy scalar values into
            # native Python values for serialization.
            if hasattr(value, "item"):
                value = value.item()

            learned_parameters[column] = {
                "strategy": strategy,
                "value": value,
            }

        # Save state only after successful fitting.
        self._columns = list(dataframe.columns)
        self._learned_parameters = learned_parameters
        self._is_fitted = True

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> CleaningResult:
        """
        Apply previously learned cleaning parameters.
        """

        self._validate_dataframe(dataframe)

        if not self._is_fitted:
            raise RuntimeError(
                "DataCleaner must be fitted before transformation."
            )

        if list(dataframe.columns) != self._columns:
            raise ValueError(
                "Input columns must match the fitted "
                "training DataFrame columns and order."
            )

        original_row_count = len(dataframe)

        missing_before = {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
            if count > 0
        }

        duplicate_count_before = int(
            dataframe.duplicated().sum()
        )

        # Work on an independent copy.
        cleaned = dataframe.copy(deep=True)

        # Apply learned cleaning parameters.
        for column, parameters in (
            self._learned_parameters.items()
        ):

            strategy = parameters["strategy"]
            value = parameters["value"]

            if strategy in (
                "mean",
                "median",
                "mode",
                "constant",
            ):
                cleaned[column] = cleaned[column].fillna(
                    value
                )

            elif strategy == "drop_rows":
                cleaned = cleaned.dropna(
                    subset=[column]
                )

        rows_removed_missing = (
            original_row_count - len(cleaned)
        )

        rows_before_duplicate_removal = len(cleaned)

        # Apply duplicate removal.
        if self.rules.remove_duplicates:

            cleaned = cleaned.drop_duplicates(
                keep=self.rules.duplicate_keep
            )

        duplicates_removed = (
            rows_before_duplicate_removal - len(cleaned)
        )

        missing_after = {
            column: int(count)
            for column, count in cleaned.isna().sum().items()
            if count > 0
        }

        report = CleaningReport(
            original_row_count=original_row_count,
            final_row_count=len(cleaned),
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            duplicate_rows_before=duplicate_count_before,
            duplicates_removed=duplicates_removed,
            rows_removed_missing=rows_removed_missing,
        )

        return CleaningResult(
            dataframe=cleaned,
            report=report,
        )

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> CleaningResult:
        """
        Learn cleaning parameters and transform training data.
        """

        self.fit(dataframe)

        return self.transform(dataframe)

    def get_state(self) -> dict:
        """
        Return a copy of the learned cleaning state.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "DataCleaner must be fitted before retrieving state."
            )

        return {
            "columns": list(self._columns),
            "learned_parameters": {
                column: parameters.copy()
                for column, parameters
                in self._learned_parameters.items()
            },
        }


def clean_dataset(
    dataframe: pd.DataFrame,
    rules: CleaningRules | None = None,
) -> CleaningResult:
    """
    Backward-compatible cleaning function.

    Uses DataCleaner internally.
    """

    cleaner = DataCleaner(rules)

    return cleaner.fit_transform(dataframe)
