
from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.model_selection import train_test_split


@dataclass
class ModelTrainingRules:
    """
    Configurable rules for model training.
    """

    task_type: str
    model_type: str

    test_size: float = 0.2
    random_state: int = 42

    n_estimators: int = 100
    max_iter: int = 1000


@dataclass
class ModelTrainingReport:
    """
    Contains information about the model training process.
    """

    task_type: str = ""
    model_type: str = ""

    total_rows: int = 0
    training_rows: int = 0
    testing_rows: int = 0

    feature_columns: list[str] = field(
        default_factory=list
    )

    target_classes: list[str] = field(
        default_factory=list
    )


@dataclass
class ModelTrainingResult:
    """
    Contains the trained model, dataset splits
    and training report.
    """

    model: object

    X_train: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_test: pd.Series

    report: ModelTrainingReport


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    rules: ModelTrainingRules,
) -> ModelTrainingResult:
    """
    Train a machine learning model.

    Supported operations:
    - Input validation
    - Training and testing split
    - Model selection
    - Model training
    - Training report generation

    The original input datasets are never modified.
    """

    # Validate feature dataset.
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Expected X to be a Pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Feature dataset cannot be empty."
        )

    if not X.columns.is_unique:
        raise ValueError(
            "Feature dataset must have unique column names."
        )

    # Validate target dataset.
    if not isinstance(y, pd.Series):
        raise TypeError(
            "Expected y to be a Pandas Series."
        )

    if y.empty:
        raise ValueError(
            "Target dataset cannot be empty."
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature and target datasets must have "
            "the same number of rows."
        )

    if X.index.has_duplicates or y.index.has_duplicates:
        raise ValueError(
            "Feature and target indexes must be unique."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "Feature and target indexes must match."
        )

    if X.isnull().any().any():
        raise ValueError(
            "Feature dataset contains missing values. "
            "Clean the data before training."
        )

    if y.isnull().any():
        raise ValueError(
            "Target dataset contains missing values."
        )

    # Validate numeric features.
    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError(
            "All features must be numeric. "
            "Encode categorical features before training."
        )

    # Validate training configuration.
    if not isinstance(rules, ModelTrainingRules):
        raise TypeError(
            "Expected ModelTrainingRules configuration."
        )

    if rules.task_type not in (
        "classification",
        "regression",
    ):
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    supported_models = {
        "classification": [
            "logistic_regression",
            "random_forest",
        ],
        "regression": [
            "linear_regression",
            "random_forest",
        ],
    }

    if rules.model_type not in supported_models[rules.task_type]:
        raise ValueError(
            f"Model '{rules.model_type}' is not supported "
            f"for {rules.task_type}."
        )

    if not 0 < rules.test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not isinstance(rules.random_state, int):
        raise TypeError(
            "random_state must be an integer."
        )

    if rules.n_estimators < 1:
        raise ValueError(
            "n_estimators must be at least 1."
        )

    if rules.max_iter < 1:
        raise ValueError(
            "max_iter must be at least 1."
        )

    if len(X) < 2:
        raise ValueError(
            "At least two rows are required for training."
        )

    # Validate target configuration.
    if rules.task_type == "classification":
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        class_counts = y.value_counts()

        if class_counts.min() < 2:
            raise ValueError(
                "Each target class must contain at least "
                "two rows for stratified splitting."
            )

        test_rows = max(
            1,
            int(len(X) * rules.test_size + 0.999999),
        )

        train_rows = len(X) - test_rows

        if test_rows < y.nunique() or train_rows < y.nunique():
            raise ValueError(
                "Not enough rows for stratified training "
                "and testing splits."
            )

    # Split dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X.copy(deep=True),
        y.copy(deep=True),
        test_size=rules.test_size,
        random_state=rules.random_state,
        stratify=y if rules.task_type == "classification" else None,
    )

    # Select machine learning model.
    if rules.task_type == "classification":

        if rules.model_type == "logistic_regression":
            model = LogisticRegression(
                max_iter=rules.max_iter,
                random_state=rules.random_state,
            )

        else:
            model = RandomForestClassifier(
                n_estimators=rules.n_estimators,
                random_state=rules.random_state,
            )

    else:

        if rules.model_type == "linear_regression":
            model = LinearRegression()

        else:
            model = RandomForestRegressor(
                n_estimators=rules.n_estimators,
                random_state=rules.random_state,
            )

    # Train model.
    model.fit(X_train, y_train)

    # Generate training report.
    report = ModelTrainingReport(
        task_type=rules.task_type,
        model_type=rules.model_type,
        total_rows=len(X),
        training_rows=len(X_train),
        testing_rows=len(X_test),
        feature_columns=list(X.columns),
        target_classes=(
            [str(value) for value in y.unique()]
            if rules.task_type == "classification"
            else []
        ),
    )

    return ModelTrainingResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        report=report,
    )

