
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from evolyrn.data.preprocessing.data_cleaner import (
    DataCleaner,
    CleaningRules,
    CleaningReport,
)

from evolyrn.data.preprocessing.data_encoder import (
    DataEncoder,
    EncodingRules,
    EncodingReport,
)

from evolyrn.data.preprocessing.data_scaler import (
    DataScaler,
    ScalingRules,
    ScalingReport,
)

from evolyrn.preparation.feature_preparer import (
    FeaturePreparationRules,
    FeaturePreparationReport,
    prepare_features,
)


@dataclass
class PreprocessingPipelineReport:
    """
    Contains the complete preprocessing pipeline report.
    """

    input_rows: int = 0
    output_rows: int = 0

    input_columns: list[str] = None
    output_columns: list[str] = None

    cleaning_report: CleaningReport | None = None

    feature_preparation_report: (
        FeaturePreparationReport | None
    ) = None

    encoding_report: EncodingReport | None = None

    scaling_report: ScalingReport | None = None

    def __post_init__(self):
        if self.input_columns is None:
            self.input_columns = []

        if self.output_columns is None:
            self.output_columns = []


@dataclass
class PreprocessingPipelineResult:
    """
    Contains the final prepared features, target
    and complete preprocessing report.
    """

    X: pd.DataFrame
    y: pd.Series

    report: PreprocessingPipelineReport


class UnifiedPreprocessingPipeline:
    """
    Unified preprocessing orchestration engine.

    Integrates:
    - DataCleaner
    - FeaturePreparer
    - DataEncoder
    - DataScaler

    Lifecycle:
    - fit()
    - transform()
    - fit_transform()
    - get_state()

    Training workflow:
    1. Learn cleaning parameters.
    2. Clean the training dataset.
    3. Separate features and target.
    4. Learn encoding mappings.
    5. Encode training features.
    6. Learn scaling parameters.

    Transformation workflow:
    1. Apply learned cleaning parameters.
    2. Separate features and target.
    3. Apply learned encoding mappings.
    4. Apply learned scaling parameters.

    The target is never passed through the encoder
    or scaler.

    The original DataFrame is never modified.
    """

    def __init__(
        self,
        feature_rules: FeaturePreparationRules,
        cleaning_rules: CleaningRules | None = None,
        encoding_rules: EncodingRules | None = None,
        scaling_rules: ScalingRules | None = None,
    ):

        # Validate feature preparation configuration.
        if not isinstance(
            feature_rules,
            FeaturePreparationRules,
        ):
            raise TypeError(
                "Expected FeaturePreparationRules configuration."
            )

        # Apply default configurations.
        if cleaning_rules is None:
            cleaning_rules = CleaningRules()

        if encoding_rules is None:
            encoding_rules = EncodingRules()

        if scaling_rules is None:
            scaling_rules = ScalingRules()

        # Validate component configurations.
        if not isinstance(cleaning_rules, CleaningRules):
            raise TypeError(
                "Expected CleaningRules configuration."
            )

        if not isinstance(encoding_rules, EncodingRules):
            raise TypeError(
                "Expected EncodingRules configuration."
            )

        if not isinstance(scaling_rules, ScalingRules):
            raise TypeError(
                "Expected ScalingRules configuration."
            )

        # Preserve independent configurations.
        self.feature_rules = deepcopy(feature_rules)

        self.cleaning_rules = deepcopy(cleaning_rules)

        self.encoding_rules = deepcopy(encoding_rules)

        self.scaling_rules = deepcopy(scaling_rules)

        # Initialize preprocessing components.
        self._cleaner = None

        self._encoder = None

        self._scaler = None

        # Initialize pipeline state.
        self._is_fitted = False

        self._training_columns = []

        self._training_feature_columns = []

        self._training_encoded_columns = []

        self._training_target_column = None

    # ==================================================
    # INPUT VALIDATION
    # ==================================================

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate input DataFrame.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Expected a Pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot process an empty DataFrame."
            )

        if not dataframe.columns.is_unique:
            raise ValueError(
                "DataFrame must have unique column names."
            )

    # ==================================================
    # SCHEMA VALIDATION
    # ==================================================

    def _validate_training_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Ensure validation or inference data has the
        same columns and order as training data.
        """

        if list(dataframe.columns) != self._training_columns:
            raise ValueError(
                "Input columns must match the fitted "
                "training DataFrame columns and order."
            )

    # ==================================================
    # CONFIGURATION VALIDATION
    # ==================================================

    def _validate_configuration(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Validate feature selection and preprocessing rules.
        """

        # Validate feature selection.
        feature_result = prepare_features(
            dataframe,
            self.feature_rules,
        )

        selected_features = list(
            feature_result.X.columns
        )

        target_column = self.feature_rules.target_column

        # Target values must be available.
        if dataframe[target_column].isna().any():
            raise ValueError(
                "Target column contains missing values. "
                "Handle missing target values before training."
            )

        # Cleaning rules must reference selected features.
        invalid_cleaning_columns = (
            set(
                self.cleaning_rules.missing_value_strategies
            )
            - set(selected_features)
        )

        if invalid_cleaning_columns:
            raise ValueError(
                "Cleaning rules reference columns outside "
                "the selected features: "
                f"{sorted(invalid_cleaning_columns)}"
            )

        # Encoding rules must reference selected features.
        invalid_encoding_columns = (
            set(self.encoding_rules.strategies)
            - set(selected_features)
        )

        if invalid_encoding_columns:
            raise ValueError(
                "Encoding rules reference columns outside "
                "the selected features: "
                f"{sorted(invalid_encoding_columns)}"
            )

        return selected_features

    # ==================================================
    # INTERNAL TRANSFORMATION
    # ==================================================

    def _transform_stages(
        self,
        dataframe: pd.DataFrame,
    ) -> PreprocessingPipelineResult:
        """
        Execute all fitted preprocessing stages.

        This method never learns new parameters.
        """

        # STEP 1: Apply learned cleaning parameters.
        cleaning_result = self._cleaner.transform(
            dataframe
        )

        cleaned_dataframe = cleaning_result.dataframe

        if cleaned_dataframe.empty:
            raise ValueError(
                "No rows remain after data cleaning."
            )

        # STEP 2: Validate target after cleaning.
        target_column = self._training_target_column

        if cleaned_dataframe[target_column].isna().any():
            raise ValueError(
                "Target column contains missing values "
                "after cleaning."
            )

        # STEP 3: Separate features and target.
        feature_result = prepare_features(
            cleaned_dataframe,
            self.feature_rules,
        )

        X = feature_result.X

        y = feature_result.y

        if X.empty:
            raise ValueError(
                "No features available after preparation."
            )

        # STEP 4: Apply learned encoding mappings.
        encoding_result = self._encoder.transform(X)

        encoded_features = encoding_result.dataframe

        # STEP 5: Verify encoded feature schema.
        if (
            list(encoded_features.columns)
            != self._training_encoded_columns
        ):
            raise ValueError(
                "Encoded feature columns do not match "
                "the training feature schema."
            )

        # STEP 6: Apply learned scaling parameters.
        scaling_result = self._scaler.transform(
            encoded_features
        )

        prepared_features = scaling_result.dataframe

        # STEP 7: Verify row alignment.
        if not prepared_features.index.equals(y.index):
            raise RuntimeError(
                "Feature and target row alignment was lost."
            )

        # STEP 8: Generate complete pipeline report.
        report = PreprocessingPipelineReport(
            input_rows=len(dataframe),
            output_rows=len(prepared_features),
            input_columns=list(dataframe.columns),
            output_columns=list(
                prepared_features.columns
            ),
            cleaning_report=cleaning_result.report,
            feature_preparation_report=feature_result.report,
            encoding_report=encoding_result.report,
            scaling_report=scaling_result.report,
        )

        return PreprocessingPipelineResult(
            X=prepared_features,
            y=y,
            report=report,
        )

    # ==================================================
    # FIT: LEARN PREPROCESSING PARAMETERS
    # ==================================================

    def fit(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Learn all preprocessing parameters from training data.

        Does not modify the original DataFrame.
        """

        self._validate_dataframe(dataframe)

        selected_features = self._validate_configuration(
            dataframe
        )

        target_column = self.feature_rules.target_column

        # Create fresh components locally.
        # Existing fitted state is not overwritten
        # unless the complete fitting process succeeds.

        cleaner = DataCleaner(
            deepcopy(self.cleaning_rules)
        )

        encoder = DataEncoder(
            deepcopy(self.encoding_rules)
        )

        scaler = DataScaler(
            deepcopy(self.scaling_rules)
        )

        # STEP 1: Learn cleaning parameters.
        cleaner.fit(dataframe)

        # STEP 2: Clean training data.
        cleaning_result = cleaner.transform(
            dataframe
        )

        cleaned_dataframe = cleaning_result.dataframe

        if cleaned_dataframe.empty:
            raise ValueError(
                "No training rows remain after data cleaning."
            )

        # STEP 3: Validate target after cleaning.
        if cleaned_dataframe[target_column].isna().any():
            raise ValueError(
                "Target column contains missing values "
                "after cleaning."
            )

        # STEP 4: Separate features and target.
        feature_result = prepare_features(
            cleaned_dataframe,
            self.feature_rules,
        )

        X = feature_result.X

        if X.empty:
            raise ValueError(
                "No training features available."
            )

        # STEP 5: Learn encoding mappings.
        encoder.fit(X)

        # STEP 6: Encode training features.
        encoding_result = encoder.transform(X)

        encoded_features = encoding_result.dataframe

        # STEP 7: Validate scaling configuration.
        missing_scaling_columns = (
            set(self.scaling_rules.strategies)
            - set(encoded_features.columns)
        )

        if missing_scaling_columns:
            raise ValueError(
                "Scaling rules reference columns unavailable "
                "after encoding: "
                f"{sorted(missing_scaling_columns)}"
            )

        # STEP 8: Learn scaling parameters.
        scaler.fit(encoded_features)

        # STEP 9: Commit learned state.
        # Only update pipeline state after all stages succeed.

        self._cleaner = cleaner

        self._encoder = encoder

        self._scaler = scaler

        self._training_columns = list(
            dataframe.columns
        )

        self._training_feature_columns = list(
            selected_features
        )

        self._training_encoded_columns = list(
            encoded_features.columns
        )

        self._training_target_column = target_column

        self._is_fitted = True

        return self

    # ==================================================
    # TRANSFORM: REUSE LEARNED PARAMETERS
    # ==================================================

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> PreprocessingPipelineResult:
        """
        Transform validation or inference data using
        previously learned training parameters.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before transform()."
            )

        self._validate_dataframe(dataframe)

        self._validate_training_schema(dataframe)

        return self._transform_stages(dataframe)

    # ==================================================
    # FIT_TRANSFORM
    # ==================================================

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> PreprocessingPipelineResult:
        """
        Learn preprocessing parameters and transform
        the training dataset.
        """

        self.fit(dataframe)

        return self._transform_stages(dataframe)

    # ==================================================
    # STATE RETRIEVAL
    # ==================================================

    def get_state(self) -> dict:
        """
        Return the complete learned preprocessing state.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before retrieving state."
            )

        return {
            "fitted": self._is_fitted,

            "training_columns": list(
                self._training_columns
            ),

            "training_feature_columns": list(
                self._training_feature_columns
            ),

            "training_encoded_columns": list(
                self._training_encoded_columns
            ),

            "target_column": self._training_target_column,

            "cleaning_state": deepcopy(
                self._cleaner.get_state()
            ),

            "encoding_state": deepcopy(
                self._encoder.get_state()
            ),

            "scaling_state": deepcopy(
                self._scaler.get_state()
            ),
        }
