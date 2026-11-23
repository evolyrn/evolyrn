
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


def clean_dataset(
    dataframe: pd.DataFrame,
    rules: CleaningRules | None = None,
) -> CleaningResult:
    """
    Clean a Pandas DataFrame using configurable strategies.

    Supported missing-value strategies:
    - mean
    - median
    - mode
    - constant
    - drop_rows

    The original DataFrame is never modified.
    """

    # Validate input DataFrame.
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    # Apply default configuration.
    if rules is None:
        rules = CleaningRules()

    if not isinstance(rules, CleaningRules):
        raise TypeError("Expected CleaningRules configuration.")

    if rules.duplicate_keep not in ("first", "last"):
        raise ValueError(
            "duplicate_keep must be either 'first' or 'last'."
        )

    valid_strategies = {
        "mean",
        "median",
        "mode",
        "constant",
        "drop_rows",
    }

    # Validate cleaning configurations.
    for column, strategy in rules.missing_value_strategies.items():

        if column not in dataframe.columns:
            raise ValueError(
                f"Cleaning rule references unknown column '{column}'."
            )

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported cleaning strategy '{strategy}'."
            )

        if strategy in ("mean", "median"):

            if dataframe[column].isna().all():
                raise ValueError(
                    f"Cannot calculate {strategy} for column '{column}' "
                    "because all values are missing."
                )

            if not is_numeric_dtype(dataframe[column]):
                raise TypeError(
                    f"Strategy '{strategy}' requires a numeric column: '{column}'."
                )

        if (
            strategy == "constant"
            and column not in rules.fill_values
        ):
            raise ValueError(
                f"Constant fill value not provided for '{column}'."
            )

    # Capture original dataset statistics.
    original_row_count = len(dataframe)

    missing_before = {
        column: int(count)
        for column, count in dataframe.isna().sum().items()
        if count > 0
    }

    duplicate_count_before = int(
        dataframe.duplicated().sum()
    )

    # Create an independent copy.
    cleaned = dataframe.copy(deep=True)

    # Handle missing values.
    for column, strategy in rules.missing_value_strategies.items():

        if strategy == "mean":

            value = cleaned[column].mean()
            cleaned[column] = cleaned[column].fillna(value)

        elif strategy == "median":

            value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(value)

        elif strategy == "mode":

            modes = cleaned[column].mode()

            if modes.empty:
                raise ValueError(
                    f"Cannot calculate mode for column '{column}'."
                )

            cleaned[column] = cleaned[column].fillna(
                modes.iloc[0]
            )

        elif strategy == "constant":

            cleaned[column] = cleaned[column].fillna(
                rules.fill_values[column]
            )

        elif strategy == "drop_rows":

            cleaned = cleaned.dropna(
                subset=[column]
            )

    # Calculate rows removed due to missing values.
    rows_removed_missing = (
        original_row_count - len(cleaned)
    )

    # Capture row count before duplicate removal.
    rows_before_duplicate_removal = len(cleaned)

    # Remove duplicate records.
    if rules.remove_duplicates:

        cleaned = cleaned.drop_duplicates(
            keep=rules.duplicate_keep
        )

    # Calculate actual duplicate removals.
    duplicates_removed = (
        rows_before_duplicate_removal - len(cleaned)
    )

    # Capture remaining missing values.
    missing_after = {
        column: int(count)
        for column, count in cleaned.isna().sum().items()
        if count > 0
    }

    # Generate cleaning report.
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
