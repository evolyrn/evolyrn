
import pandas as pd
import pytest

from evolyrn.training.model_trainer import (
    ModelTrainingRules,
    train_model,
)


def create_classification_data():
    X = pd.DataFrame({
        "age": range(40),
        "salary": range(30000, 70000, 1000),
    })

    y = pd.Series([0, 1] * 20, name="purchased")

    return X, y


def create_regression_data():
    X = pd.DataFrame({
        "area": range(1, 41),
        "bedrooms": [2, 3, 4, 5] * 10,
    })

    y = pd.Series(
        [100000 + value * 5000 for value in range(40)],
        name="price",
    )

    return X, y


def test_random_forest_classification_training():
    X, y = create_classification_data()

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="random_forest",
    )

    result = train_model(X, y, rules)

    assert result.model is not None
    assert len(result.X_train) == 32
    assert len(result.X_test) == 8
    assert result.report.task_type == "classification"
    assert result.report.model_type == "random_forest"


def test_logistic_regression_training():
    X, y = create_classification_data()

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="logistic_regression",
    )

    result = train_model(X, y, rules)

    assert result.model is not None
    assert result.report.training_rows == 32


def test_random_forest_regression_training():
    X, y = create_regression_data()

    rules = ModelTrainingRules(
        task_type="regression",
        model_type="random_forest",
    )

    result = train_model(X, y, rules)

    assert result.model is not None
    assert len(result.X_train) == 32
    assert len(result.X_test) == 8


def test_linear_regression_training():
    X, y = create_regression_data()

    rules = ModelTrainingRules(
        task_type="regression",
        model_type="linear_regression",
    )

    result = train_model(X, y, rules)

    assert result.model is not None
    assert result.report.task_type == "regression"


def test_invalid_task_type():
    X, y = create_classification_data()

    rules = ModelTrainingRules(
        task_type="invalid",
        model_type="random_forest",
    )

    with pytest.raises(ValueError):
        train_model(X, y, rules)


def test_invalid_model_type():
    X, y = create_classification_data()

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="linear_regression",
    )

    with pytest.raises(ValueError):
        train_model(X, y, rules)


def test_mismatched_row_count():
    X, y = create_classification_data()

    y = y.iloc[:-1]

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="random_forest",
    )

    with pytest.raises(ValueError):
        train_model(X, y, rules)


def test_missing_feature_values():
    X, y = create_classification_data()

    X.loc[0, "age"] = None

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="random_forest",
    )

    with pytest.raises(ValueError):
        train_model(X, y, rules)


def test_original_data_is_not_modified():
    X, y = create_classification_data()

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="random_forest",
    )

    train_model(X, y, rules)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_training_report_contains_feature_columns():
    X, y = create_classification_data()

    rules = ModelTrainingRules(
        task_type="classification",
        model_type="random_forest",
    )

    result = train_model(X, y, rules)

    assert result.report.feature_columns == [
        "age",
        "salary",
    ]

    assert result.report.target_classes == ["0", "1"]

