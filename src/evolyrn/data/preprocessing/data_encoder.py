
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EncodingRules:
    """
    Configurable rules for dataset encoding.
    """

    strategies: dict[str, str] = field(
        default_factory=dict
    )

    ordinal_categories: dict[str, list] = field(
        default_factory=dict
    )

    drop_original: bool = True


@dataclass
class EncodingReport:
    """
    Contains the complete dataset encoding results.
    """

    original_columns: list[str] = field(
        default_factory=list
    )

    encoded_columns: dict[str, list[str]] = field(
        default_factory=dict
    )

    strategies_applied: dict[str, str] = field(
        default_factory=dict
    )

    rows_processed: int = 0


@dataclass
class EncodingResult:
    """
    Contains the encoded dataset and encoding report.
    """

    dataframe: pd.DataFrame
    report: EncodingReport

    category_mappings: dict[str, dict] = field(
        default_factory=dict
    )


def encode_dataset(
    dataframe: pd.DataFrame,
    rules: EncodingRules | None = None,
) -> EncodingResult:
    """
    Encode categorical columns using configurable strategies.

    Supported encoding strategies:
    - one_hot
    - label
    - ordinal

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
        rules = EncodingRules()

    if not isinstance(rules, EncodingRules):
        raise TypeError(
            "Expected EncodingRules configuration."
        )

    valid_strategies = {
        "one_hot",
        "label",
        "ordinal",
    }

    # Validate encoding configurations.
    for column, strategy in rules.strategies.items():

        if column not in dataframe.columns:
            raise ValueError(
                f"Encoding rule references unknown column '{column}'."
            )

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported encoding strategy '{strategy}'."
            )

        if dataframe[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values. "
                "Clean missing values before encoding."
            )

    # Validate ordinal category configurations.
    for column, categories in rules.ordinal_categories.items():

        if column not in rules.strategies:
            raise ValueError(
                f"Ordinal categories provided for '{column}' "
                "without an encoding strategy."
            )

        if rules.strategies[column] != "ordinal":
            raise ValueError(
                f"Ordinal categories can only be used "
                f"with ordinal encoding: '{column}'."
            )

        if not categories:
            raise ValueError(
                f"Ordinal categories cannot be empty for '{column}'."
            )

        if len(categories) != len(set(categories)):
            raise ValueError(
                f"Duplicate ordinal categories found for '{column}'."
            )

    # Capture original dataset information.
    original_columns = list(dataframe.columns)

    # Create an independent copy.
    encoded = dataframe.copy(deep=True)

    # Initialize reports and mappings.
    encoded_columns = {}
    strategies_applied = {}
    category_mappings = {}

    # Apply encoding strategies.
    for column, strategy in rules.strategies.items():

        values = list(pd.unique(encoded[column]))

        # One-Hot Encoding.
        if strategy == "one_hot":

            mapping = {}
            new_columns = []

            for index, category in enumerate(values):

                new_column = f"{column}_{category}"

                if (
                    new_column in encoded.columns
                    or new_column in new_columns
                ):
                    raise ValueError(
                        f"Generated column name '{new_column}' "
                        "already exists."
                    )

                encoded[new_column] = (
                    encoded[column] == category
                ).astype("int8")

                mapping[category] = new_column
                new_columns.append(new_column)

            if rules.drop_original:
                encoded = encoded.drop(columns=[column])

            category_mappings[column] = mapping
            encoded_columns[column] = new_columns

        # Label Encoding.
        elif strategy == "label":

            mapping = {
                category: index
                for index, category in enumerate(values)
            }

            encoded[column] = encoded[column].map(mapping)

            category_mappings[column] = mapping
            encoded_columns[column] = [column]

        # Ordinal Encoding.
        elif strategy == "ordinal":

            categories = rules.ordinal_categories.get(column)

            if categories is None:
                raise ValueError(
                    f"Ordinal category order not provided "
                    f"for '{column}'."
                )

            mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            unknown_values = [
                value for value in values
                if value not in mapping
            ]

            if unknown_values:
                raise ValueError(
                    f"Unknown ordinal categories in '{column}': "
                    f"{unknown_values}"
                )

            encoded[column] = encoded[column].map(mapping)

            category_mappings[column] = mapping
            encoded_columns[column] = [column]

        strategies_applied[column] = strategy

    # Generate encoding report.
    report = EncodingReport(
        original_columns=original_columns,
        encoded_columns=encoded_columns,
        strategies_applied=strategies_applied,
        rows_processed=len(encoded),
    )

    return EncodingResult(
        dataframe=encoded,
        report=report,
        category_mappings=category_mappings,
    )

