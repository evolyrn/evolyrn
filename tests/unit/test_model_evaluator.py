
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from evolyrn.evaluation.model_evaluator import (
    ModelEvaluationRules,
    evaluate_model,
)


def create_classification_data():
    X = pd.DataFrame({
        "age": range(40),
        "salary": range(30000, 70000, 1000),
    })

    y = pd.Series([0, 1] * 20, name="purchased")

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(X, y)

    return model, X, y


def create_regression_data():
    X = pd.DataFrame({
        "area": range(1, 41),
        "bedrooms": [2, 3, 4, 5] * 10,
    })

    y = pd.Series(
        [100000 + value * 5000 for value in range(40)],
        name="price",
    )

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42,
    )

    model.fit(X, y)

    return model, X, y


def test_classification_evaluation():
    model, X, y = create_classification_data()

    rules = ModelEvaluationRules(
        task_type="classification",
    )

    result = evaluate_model(model, X, y, rules)

    assert result.report.task_type == "classification"
    assert result.report.samples_evaluated == 40

    assert result.report.accuracy is not None
    assert result.report.precision is not None
    assert result.report.recall is not None
    assert result.report.f1_score is not None

    assert len(result.predictions) == 40


def test_regression_evaluation():
    model, X, y = create_regression_data()

    rules = ModelEvaluationRules(
        task_type="regression",
    )

    result = evaluate_model(model, X, y, rules)

    assert result.report.task_type == "regression"
    assert result.report.samples_evaluated == 40

    assert result.report.mae is not None
    assert result.report.mse is not None
    assert result.report.rmse is not None
    assert result.report.r2_score is not None

    assert len(result.predictions) == 40


def test_classification_metrics_are_in_valid_range():
    model, X, y = create_classification_data()

    result = evaluate_model(
        model,
        X,
        y,
        ModelEvaluationRules(task_type="classification"),
    )

    assert 0 <= result.report.accuracy <= 1
    assert 0 <= result.report.precision <= 1
    assert 0 <= result.report.recall <= 1
    assert 0 <= result.report.f1_score <= 1


def test_regression_rmse_matches_mse():
    model, X, y = create_regression_data()

    result = evaluate_model(
        model,
        X,
        y,
        ModelEvaluationRules(task_type="regression"),
    )

    assert result.report.rmse == pytest.approx(
        np.sqrt(result.report.mse)
    )


def test_invalid_task_type():
    model, X, y = create_classification_data()

    rules = ModelEvaluationRules(
        task_type="invalid",
    )

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, rules)


def test_invalid_model():
    _, X, y = create_classification_data()

    with pytest.raises(TypeError):
        evaluate_model(
            object(),
            X,
            y,
            ModelEvaluationRules(task_type="classification"),
        )


def test_mismatched_row_count():
    model, X, y = create_classification_data()

    y = y.iloc[:-1]

    with pytest.raises(ValueError):
        evaluate_model(
            model,
            X,
            y,
            ModelEvaluationRules(task_type="classification"),
        )


def test_missing_feature_values():
    model, X, y = create_classification_data()

    X.loc[0, "age"] = np.nan

    with pytest.raises(ValueError):
        evaluate_model(
            model,
            X,
            y,
            ModelEvaluationRules(task_type="classification"),
        )


def test_missing_target_values():
    model, X, y = create_classification_data()

    y.iloc[0] = np.nan

    with pytest.raises(ValueError):
        evaluate_model(
            model,
            X,
            y,
            ModelEvaluationRules(task_type="classification"),
        )


def test_original_data_is_not_modified():
    model, X, y = create_classification_data()

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    evaluate_model(
        model,
        X,
        y,
        ModelEvaluationRules(task_type="classification"),
    )

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_invalid_classification_average():
    model, X, y = create_classification_data()

    rules = ModelEvaluationRules(
        task_type="classification",
        average="invalid",
    )

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, rules)


def test_non_numeric_regression_target():
    model, X, _ = create_regression_data()

    y = pd.Series(["low"] * 40)

    rules = ModelEvaluationRules(
        task_type="regression",
    )

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, rules)
