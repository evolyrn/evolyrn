
import pandas as pd

from sklearn.datasets import make_classification, make_regression

from evolyrn.training.model_trainer import (
    ModelTrainingRules,
    train_model,
)

from evolyrn.evaluation.model_evaluator import (
    ModelEvaluationRules,
    evaluate_model,
)

from evolyrn.selection.model_selector import (
    ModelSelectionCandidate,
    ModelSelectionRules,
    select_best_model,
)


def run_classification_test():

    print("\n========== CLASSIFICATION TEST ==========")

    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42,
    )

    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(6)],
    )

    y = pd.Series(y, index=X.index, name="target")

    candidates = []

    models = [
        "logistic_regression",
        "random_forest",
    ]

    for model_type in models:

        training_result = train_model(
            X,
            y,
            ModelTrainingRules(
                task_type="classification",
                model_type=model_type,
            ),
        )

        evaluation_result = evaluate_model(
            training_result.model,
            training_result.X_test,
            training_result.y_test,
            ModelEvaluationRules(
                task_type="classification",
            ),
        )

        candidates.append(
            ModelSelectionCandidate(
                model_type=model_type,
                model=training_result.model,
                evaluation_report=evaluation_result.report,
            )
        )

        print(
            f"{model_type}: "
            f"F1={evaluation_result.report.f1_score:.4f}, "
            f"Accuracy={evaluation_result.report.accuracy:.4f}"
        )

    selection_result = select_best_model(
        candidates,
        ModelSelectionRules(
            task_type="classification",
        ),
    )

    print("\nSELECTED CLASSIFICATION MODEL:")
    print(selection_result.selected_model_type)

    print("Selection score:", selection_result.selected_score)


def run_regression_test():

    print("\n========== REGRESSION TEST ==========")

    X, y = make_regression(
        n_samples=300,
        n_features=6,
        n_informative=4,
        noise=15,
        random_state=42,
    )

    X = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(6)],
    )

    y = pd.Series(y, index=X.index, name="target")

    candidates = []

    models = [
        "linear_regression",
        "random_forest",
    ]

    for model_type in models:

        training_result = train_model(
            X,
            y,
            ModelTrainingRules(
                task_type="regression",
                model_type=model_type,
            ),
        )

        evaluation_result = evaluate_model(
            training_result.model,
            training_result.X_test,
            training_result.y_test,
            ModelEvaluationRules(
                task_type="regression",
            ),
        )

        candidates.append(
            ModelSelectionCandidate(
                model_type=model_type,
                model=training_result.model,
                evaluation_report=evaluation_result.report,
            )
        )

        print(
            f"{model_type}: "
            f"RMSE={evaluation_result.report.rmse:.4f}, "
            f"R2={evaluation_result.report.r2_score:.4f}"
        )

    selection_result = select_best_model(
        candidates,
        ModelSelectionRules(
            task_type="regression",
        ),
    )

    print("\nSELECTED REGRESSION MODEL:")
    print(selection_result.selected_model_type)

    print("Selection score:", selection_result.selected_score)


if __name__ == "__main__":

    run_classification_test()

    run_regression_test()

    print("\n========== ALL TESTS COMPLETED ==========")

