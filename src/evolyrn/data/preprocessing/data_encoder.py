
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EncodingRules:
    """Configurable rules for dataset encoding."""

    strategies: dict[str, str] = field(
        default_factory=dict
    )

    ordinal_categories: dict[str, list] = field(
        default_factory=dict
    )

    drop_original: bool = True

    handle_unknown: str = "error"


@dataclass
class EncodingReport:
    """Contains the complete dataset encoding results."""

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
    """Contains the encoded dataset and encoding report."""

    dataframe: pd.DataFrame
    report: EncodingReport

    category_mappings: dict[str, dict] = field(
        default_factory=dict
    )


class DataEncoder:
    """
    Reusable categorical encoding engine.

    Supported strategies:
    - one_hot
    - label
    - ordinal

    Lifecycle:
    - fit()
    - transform()
    - fit_transform()
    - get_state()

    The original DataFrame is never modified.
    """

    VALID_STRATEGIES = {
        "one_hot",
        "label",
        "ordinal",
    }

    VALID_UNKNOWN_POLICIES = {
        "error",
        "ignore",
    }

    def __init__(
        self,
        rules: EncodingRules | None = None,
    ):
        if rules is None:
            rules = EncodingRules()

        if not isinstance(rules, EncodingRules):
            raise TypeError(
                "Expected EncodingRules configuration."
            )

        self.rules = rules

        self._category_mappings = {}
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
                    f"Encoding rule references unknown "
                    f"column '{column}'."
                )

            if strategy not in self.VALID_STRATEGIES:
                raise ValueError(
                    f"Unsupported encoding strategy "
                    f"'{strategy}'."
                )

            if dataframe[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing "
                    "values. Clean missing values before encoding."
                )

        if (
            self.rules.handle_unknown
            not in self.VALID_UNKNOWN_POLICIES
        ):
            raise ValueError(
                "handle_unknown must be 'error' or 'ignore'."
            )

        for column, categories in (
            self.rules.ordinal_categories.items()
        ):

            if column not in self.rules.strategies:
                raise ValueError(
                    f"Ordinal categories provided for '{column}' "
                    "without an encoding strategy."
                )

            if self.rules.strategies[column] != "ordinal":
                raise ValueError(
                    "Ordinal categories can only be used "
                    f"with ordinal encoding: '{column}'."
                )

            if not categories:
                raise ValueError(
                    f"Ordinal categories cannot be empty "
                    f"for '{column}'."
                )

            if len(categories) != len(set(categories)):
                raise ValueError(
                    f"Duplicate ordinal categories found "
                    f"for '{column}'."
                )

    # --------------------------------------------------
    # FIT: LEARN ENCODING PARAMETERS
    # --------------------------------------------------

    def fit(self, dataframe: pd.DataFrame):
        """
        Learn category mappings from training data.

        Does not modify the original DataFrame.
        """

        self._validate_configuration(dataframe)

        learned_mappings = {}

        for column, strategy in self.rules.strategies.items():

            values = list(pd.unique(dataframe[column]))

            # One-Hot Encoding
            if strategy == "one_hot":

                mapping = {
                    category: f"{column}_{category}"
                    for category in values
                }

            # Label Encoding
            elif strategy == "label":

                mapping = {
                    category: index
                    for index, category in enumerate(values)
                }

            # Ordinal Encoding
            elif strategy == "ordinal":

                categories = (
                    self.rules.ordinal_categories.get(column)
                )

                if categories is None:
                    raise ValueError(
                        "Ordinal category order not provided "
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
                        f"Unknown ordinal categories in "
                        f"'{column}': {unknown_values}"
                    )

            learned_mappings[column] = mapping

        # Save learned state only after successful validation.
        self._category_mappings = learned_mappings
        self._fitted = True

        return self

    # --------------------------------------------------
    # TRANSFORM: APPLY LEARNED PARAMETERS
    # --------------------------------------------------

    def transform(self, dataframe: pd.DataFrame):
        """
        Transform data using previously learned mappings.

        Never learns new categories during transformation.
        """

        if not self._fitted:
            raise RuntimeError(
                "DataEncoder must be fitted before transform()."
            )

        self._validate_dataframe(dataframe)

        for column in self.rules.strategies:

            if column not in dataframe.columns:
                raise ValueError(
                    f"Required encoding column '{column}' "
                    "is missing from the DataFrame."
                )

            if dataframe[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing "
                    "values. Clean missing values before encoding."
                )

        encoded = dataframe.copy(deep=True)

        original_columns = list(dataframe.columns)

        encoded_columns = {}
        strategies_applied = {}

        for column, strategy in self.rules.strategies.items():

            mapping = self._category_mappings[column]

            values = list(pd.unique(encoded[column]))

            unknown_values = [
                value for value in values
                if value not in mapping
            ]

            if (
                unknown_values
                and self.rules.handle_unknown == "error"
            ):
                raise ValueError(
                    f"Unknown categories in '{column}': "
                    f"{unknown_values}"
                )

            # One-Hot Encoding
            if strategy == "one_hot":

                new_columns = []

                for category, new_column in mapping.items():

                    if (
                        new_column in encoded.columns
                        or new_column in new_columns
                    ):
                        raise ValueError(
                            f"Generated column name "
                            f"'{new_column}' already exists."
                        )

                    encoded[new_column] = (
                        encoded[column] == category
                    ).astype("int8")

                    new_columns.append(new_column)

                if self.rules.drop_original:
                    encoded = encoded.drop(
                        columns=[column]
                    )

                encoded_columns[column] = new_columns

            # Label Encoding
            elif strategy == "label":

                encoded[column] = (
                    encoded[column]
                    .map(mapping)
                    .fillna(-1)
                    .astype("int64")
                )

                encoded_columns[column] = [column]

            # Ordinal Encoding
            elif strategy == "ordinal":

                encoded[column] = (
                    encoded[column]
                    .map(mapping)
                    .fillna(-1)
                    .astype("int64")
                )

                encoded_columns[column] = [column]

            strategies_applied[column] = strategy

        report = EncodingReport(
            original_columns=original_columns,
            encoded_columns=encoded_columns,
            strategies_applied=strategies_applied,
            rows_processed=len(encoded),
        )

        return EncodingResult(
            dataframe=encoded,
            report=report,
            category_mappings=deepcopy(
                self._category_mappings
            ),
        )

    # --------------------------------------------------
    # FIT_TRANSFORM: LEARN AND TRANSFORM
    # --------------------------------------------------

    def fit_transform(self, dataframe: pd.DataFrame):
        """
        Learn mappings from training data and transform it.
        """

        self.fit(dataframe)

        return self.transform(dataframe)

    # --------------------------------------------------
    # STATE RETRIEVAL
    # --------------------------------------------------

    def get_state(self):
        """
        Return the current learned encoding state.
        """

        return {
            "fitted": self._fitted,
            "strategies": deepcopy(
                self.rules.strategies
            ),
            "category_mappings": deepcopy(
                self._category_mappings
            ),
            "handle_unknown": self.rules.handle_unknown,
        }


# ------------------------------------------------------
# BACKWARD-COMPATIBLE FUNCTION
# ------------------------------------------------------

def encode_dataset(
    dataframe: pd.DataFrame,
    rules: EncodingRules | None = None,
) -> EncodingResult:
    """
    Backward-compatible standalone encoding function.
    """

    encoder = DataEncoder(rules)

    return encoder.fit_transform(dataframe)

