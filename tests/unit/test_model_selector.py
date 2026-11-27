
import pytest

from evolyrn.evaluation.model_evaluator import (
    ModelEvaluationReport,
)

from evolyrn.selection.model_selector import (
    ModelSelectionCandidate,
    ModelSelectionRules,
    select_best_model,
)


class DummyModel:
    """Simple model used for testing."""

    def predict(self, X):
        return [0] * len(X)


def create_classification_candidate(
    model_type,
    f1_score,
    accuracy=0.85,
):
    return ModelSelectionCandidate(
        model_type=model_type,
        model=DummyModel(),
        evaluation_report=ModelEvaluationReport(
            task_type="classification",
            accuracy=accuracy,
            precision=f1_score,
            recall=f1_score,
            f1_score=f1_score,
        ),
    )


def create_regression_candidate(
    model_type,
    rmse,
    r2_score=0.80,
):
    return ModelSelectionCandidate(
        model_type=model_type,
        model=DummyModel(),
        evaluation_report=ModelEvaluationReport(
            task_type="regression",
            mae=rmse,
            mse=rmse ** 2,
            rmse=rmse,
            r2_score=r2_score,
        ),
    )


def test_select_best_classification_model():

    candidates = [
        create_classification_candidate(
            "logistic_regression",
            0.82,
        ),
        create_classification_candidate(
            "random_forest",
            0.91,
        ),
    ]

    rules = ModelSelectionRules(
        task_type="classification",
    )

    result = select_best_model(candidates, rules)

    assert result.selected_model_type == "random_forest"
    assert result.selected_score == 0.91
    assert result.metric_name == "f1_score"
    assert result.report.candidate_count == 2


def test_select_best_regression_model():

    candidates = [
        create_regression_candidate(
            "linear_regression",
            12.5,
        ),
        create_regression_candidate(
            "random_forest",
            8.2,
        ),
    ]

    rules = ModelSelectionRules(
        task_type="regression",
    )

    result = select_best_model(candidates, rules)

    assert result.selected_model_type == "random_forest"
    assert result.selected_score == 8.2
    assert result.metric_name == "rmse"


def test_classification_custom_metric():

    candidates = [
        create_classification_candidate(
            "logistic_regression",
            0.90,
            accuracy=0.95,
        ),
        create_classification_candidate(
            "random_forest",
            0.85,
            accuracy=0.98,
        ),
    ]

    rules = ModelSelectionRules(
        task_type="classification",
        metric_name="accuracy",
    )

    result = select_best_model(candidates, rules)

    assert result.selected_model_type == "random_forest"
    assert result.selected_score == 0.98


def test_regression_custom_metric():

    candidates = [
        create_regression_candidate(
            "linear_regression",
            10.0,
            r2_score=0.75,
        ),
        create_regression_candidate(
            "random_forest",
            12.0,
            r2_score=0.90,
        ),
    ]

    rules = ModelSelectionRules(
        task_type="regression",
        metric_name="r2_score",
    )

    result = select_best_model(candidates, rules)

    assert result.selected_model_type == "random_forest"
    assert result.selected_score == 0.90


def test_model_rankings():

    candidates = [
        create_classification_candidate("model_a", 0.75),
        create_classification_candidate("model_b", 0.95),
        create_classification_candidate("model_c", 0.85),
    ]

    rules = ModelSelectionRules(
        task_type="classification",
    )

    result = select_best_model(candidates, rules)

    ranking = [
        entry.model_type
        for entry in result.report.rankings
    ]

    assert ranking == [
        "model_b",
        "model_c",
        "model_a",
    ]


def test_empty_candidates():

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(ValueError):
        select_best_model([], rules)


def test_invalid_task_type():

    rules = ModelSelectionRules(
        task_type="clustering",
    )

    candidates = [
        create_classification_candidate("model_a", 0.90)
    ]

    with pytest.raises(ValueError):
        select_best_model(candidates, rules)


def test_invalid_metric():

    rules = ModelSelectionRules(
        task_type="classification",
        metric_name="rmse",
    )

    candidates = [
        create_classification_candidate("model_a", 0.90)
    ]

    with pytest.raises(ValueError):
        select_best_model(candidates, rules)


def test_duplicate_model_candidates():

    candidate = create_classification_candidate(
        "random_forest",
        0.90,
    )

    candidates = [
        candidate,
        candidate,
    ]

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(ValueError):
        select_best_model(candidates, rules)


def test_incompatible_evaluation_report():

    candidate = create_regression_candidate(
        "linear_regression",
        10.0,
    )

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(ValueError):
        select_best_model([candidate], rules)


def test_missing_evaluation_metric():

    candidate = ModelSelectionCandidate(
        model_type="random_forest",
        model=DummyModel(),
        evaluation_report=ModelEvaluationReport(
            task_type="classification",
        ),
    )

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(ValueError):
        select_best_model([candidate], rules)


def test_non_finite_evaluation_metric():

    candidate = create_classification_candidate(
        "random_forest",
        float("nan"),
    )

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(ValueError):
        select_best_model([candidate], rules)


def test_invalid_model_candidate():

    candidate = ModelSelectionCandidate(
        model_type="invalid_model",
        model=object(),
        evaluation_report=ModelEvaluationReport(
            task_type="classification",
            f1_score=0.90,
        ),
    )

    rules = ModelSelectionRules(
        task_type="classification",
    )

    with pytest.raises(TypeError):
        select_best_model([candidate], rules)

