
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeaturePreparationRules:
    """
    Configurable rules for feature preparation.
    """

    target_column: str

    feature_columns: list[str] | None = None

    exclude_columns: list[str] = field(
        default_factory=list
    )


@dataclass
class FeaturePreparationReport:
    """
    Contains the complete feature preparation results.
    """

    original_columns: list[str] = field(
        default_factory=list
    )

    feature_columns: list[str] = field(
        default_factory=list
    )

    target_column: str = ""

    excluded_columns: list[str] = field(
        default_factory=list
    )

    rows_processed: int = 0


@dataclass
class FeaturePreparationResult:
    """
    Contains prepared features, target and preparation report.
    """

    X: pd.DataFrame
    y: pd.Series

    report: FeaturePreparationReport


def prepare_features(
    dataframe: pd.DataFrame,
    rules: FeaturePreparationRules,
) -> FeaturePreparationResult:
    """
    Prepare machine learning features and target variables.

    Supported operations:
    - Target column identification
    - Feature column selection
    - Irrelevant column exclusion
    - X and y separation

    The original DataFrame is never modified.
    """

    # Validate input DataFrame.
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a Pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot prepare features from an empty DataFrame."
        )

    if not dataframe.columns.is_unique:
        raise ValueError(
            "DataFrame must have unique column names."
        )

    # Validate configuration.
    if not isinstance(rules, FeaturePreparationRules):
        raise TypeError(
            "Expected FeaturePreparationRules configuration."
        )

    if (
        not isinstance(rules.target_column, str)
        or not rules.target_column.strip()
    ):
        raise ValueError(
            "A valid target column must be provided."
        )

    if rules.target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{rules.target_column}' "
            "does not exist in the DataFrame."
        )

    # Validate excluded columns.
    if len(rules.exclude_columns) != len(
        set(rules.exclude_columns)
    ):
        raise ValueError(
            "Duplicate columns found in exclude_columns."
        )

    for column in rules.exclude_columns:

        if not isinstance(column, str):
            raise TypeError(
                "Excluded column names must be strings."
            )

        if column not in dataframe.columns:
            raise ValueError(
                f"Excluded column '{column}' "
                "does not exist in the DataFrame."
            )

    if rules.target_column in rules.exclude_columns:
        raise ValueError(
            "Target column cannot be excluded."
        )

    # Determine selected feature columns.
    if rules.feature_columns is None:

        selected_features = [
            column
            for column in dataframe.columns
            if column != rules.target_column
            and column not in rules.exclude_columns
        ]

    else:

        if not rules.feature_columns:
            raise ValueError(
                "At least one feature column must be selected."
            )

        if len(rules.feature_columns) != len(
            set(rules.feature_columns)
        ):
            raise ValueError(
                "Duplicate columns found in feature_columns."
            )

        for column in rules.feature_columns:

            if not isinstance(column, str):
                raise TypeError(
                    "Feature column names must be strings."
                )

            if column not in dataframe.columns:
                raise ValueError(
                    f"Feature column '{column}' "
                    "does not exist in the DataFrame."
                )

        if rules.target_column in rules.feature_columns:
            raise ValueError(
                "Target column cannot be used as a feature."
            )

        conflicting_columns = set(
            rules.feature_columns
        ).intersection(rules.exclude_columns)

        if conflicting_columns:
            raise ValueError(
                "Feature columns cannot also be excluded: "
                f"{sorted(conflicting_columns)}"
            )

        selected_features = list(
            rules.feature_columns
        )

    # Ensure at least one feature is selected.
    if not selected_features:
        raise ValueError(
            "No feature columns available for preparation."
        )

    # Capture original dataset information.
    original_columns = list(dataframe.columns)

    # Create independent feature and target datasets.
    X = dataframe.loc[:, selected_features].copy(deep=True)

    y = dataframe.loc[:, rules.target_column].copy(deep=True)

    # Identify excluded columns.
    excluded_columns = [
        column
        for column in original_columns
        if column != rules.target_column
        and column not in selected_features
    ]

    # Generate preparation report.
    report = FeaturePreparationReport(
        original_columns=original_columns,
        feature_columns=selected_features,
        target_column=rules.target_column,
        excluded_columns=excluded_columns,
        rows_processed=len(dataframe),
    )

    return FeaturePreparationResult(
        X=X,
        y=y,
        report=report,
    )

