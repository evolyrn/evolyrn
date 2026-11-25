
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


@dataclass
class ModelEvaluationRules:
    """
    Configurable rules for model evaluation.
    """

    task_type: str

    average: str = "weighted"
    zero_division: int = 0


@dataclass
class ModelEvaluationReport:
    """
    Contains model evaluation metrics.
    """

    task_type: str = ""
    samples_evaluated: int = 0

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    r2_score: float | None = None

    additional_information: dict = field(
        default_factory=dict
    )


@dataclass
class ModelEvaluationResult:
    """
    Contains predictions and evaluation report.
    """

    predictions: np.ndarray

    report: ModelEvaluationReport


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rules: ModelEvaluationRules,
) -> ModelEvaluationResult:
    """
    Evaluate a trained machine learning model.

    Supported operations:
    - Input validation
    - Model prediction
    - Classification metrics
    - Regression metrics
    - Evaluation report generation
    """

    # Validate model.
    if not hasattr(model, "predict") or not callable(model.predict):
        raise TypeError(
            "Model must provide a callable predict method."
        )

    # Validate feature dataset.
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "Expected X_test to be a Pandas DataFrame."
        )

    if X_test.empty:
        raise ValueError(
            "Testing feature dataset cannot be empty."
        )

    if not X_test.columns.is_unique:
        raise ValueError(
            "Testing feature dataset must have unique column names."
        )

    if X_test.isnull().any().any():
        raise ValueError(
            "Testing feature dataset contains missing values."
        )

    # Validate target dataset.
    if not isinstance(y_test, pd.Series):
        raise TypeError(
            "Expected y_test to be a Pandas Series."
        )

    if y_test.empty:
        raise ValueError(
            "Testing target dataset cannot be empty."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing features and target must have "
            "the same number of rows."
        )

    if X_test.index.has_duplicates or y_test.index.has_duplicates:
        raise ValueError(
            "Testing feature and target indexes must be unique."
        )

    if not X_test.index.equals(y_test.index):
        raise ValueError(
            "Testing feature and target indexes must match."
        )

    if y_test.isnull().any():
        raise ValueError(
            "Testing target contains missing values."
        )

    # Validate configuration.
    if not isinstance(rules, ModelEvaluationRules):
        raise TypeError(
            "Expected ModelEvaluationRules configuration."
        )

    if rules.task_type not in (
        "classification",
        "regression",
    ):
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if rules.task_type == "classification":
        if rules.average not in (
            "micro",
            "macro",
            "weighted",
        ):
            raise ValueError(
                "average must be 'micro', 'macro' or 'weighted'."
            )

        if rules.zero_division not in (0, 1):
            raise ValueError(
                "zero_division must be 0 or 1."
            )

    # Validate regression target.
    if rules.task_type == "regression":
        if not pd.api.types.is_numeric_dtype(y_test):
            raise ValueError(
                "Regression target must contain numeric values."
            )

        if not np.isfinite(y_test.to_numpy()).all():
            raise ValueError(
                "Regression target contains non-finite values."
            )

    # Generate predictions.
    predictions = np.asarray(
        model.predict(X_test)
    )

    if predictions.ndim != 1:
        raise ValueError(
            "Model predictions must be one-dimensional."
        )

    if len(predictions) != len(y_test):
        raise ValueError(
            "Number of predictions does not match "
            "the number of target values."
        )

    if pd.isna(predictions).any():
        raise ValueError(
            "Model generated missing predictions."
        )

    if rules.task_type == "regression":
        try:
            predictions = predictions.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Regression predictions must be numeric."
            ) from error

        if not np.isfinite(predictions).all():
            raise ValueError(
                "Regression predictions contain non-finite values."
            )

    # Calculate classification metrics.
    if rules.task_type == "classification":

        report = ModelEvaluationReport(
            task_type="classification",
            samples_evaluated=len(y_test),

            accuracy=float(
                accuracy_score(y_test, predictions)
            ),

            precision=float(
                precision_score(
                    y_test,
                    predictions,
                    average=rules.average,
                    zero_division=rules.zero_division,
                )
            ),

            recall=float(
                recall_score(
                    y_test,
                    predictions,
                    average=rules.average,
                    zero_division=rules.zero_division,
                )
            ),

            f1_score=float(
                f1_score(
                    y_test,
                    predictions,
                    average=rules.average,
                    zero_division=rules.zero_division,
                )
            ),
        )

    # Calculate regression metrics.
    else:

        mse = float(
            mean_squared_error(y_test, predictions)
        )

        report = ModelEvaluationReport(
            task_type="regression",
            samples_evaluated=len(y_test),

            mae=float(
                mean_absolute_error(y_test, predictions)
            ),

            mse=mse,

            rmse=float(
                np.sqrt(mse)
            ),

            r2_score=float(
                r2_score(y_test, predictions)
            ),
        )

    return ModelEvaluationResult(
        predictions=predictions,
        report=report,
    )
