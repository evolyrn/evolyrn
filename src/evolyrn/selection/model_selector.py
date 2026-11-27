
from dataclasses import dataclass, field
import math

from evolyrn.evaluation.model_evaluator import (
    ModelEvaluationReport,
)


@dataclass
class ModelSelectionRules:
    """
    Configurable rules for model selection.
    """

    task_type: str
    metric_name: str | None = None


@dataclass
class ModelSelectionCandidate:
    """
    Represents a trained model and its evaluation results.
    """

    model_type: str
    model: object
    evaluation_report: ModelEvaluationReport


@dataclass
class ModelSelectionEntry:
    """
    Contains the selection score of a candidate model.
    """

    model_type: str
    score: float


@dataclass
class ModelSelectionReport:
    """
    Contains information about the model selection process.
    """

    task_type: str = ""
    metric_name: str = ""

    selected_model_type: str = ""
    selected_score: float = 0.0

    candidate_count: int = 0

    rankings: list[ModelSelectionEntry] = field(
        default_factory=list
    )


@dataclass
class ModelSelectionResult:
    """
    Contains the selected model and selection report.
    """

    selected_model: object
    selected_model_type: str
    selected_score: float

    metric_name: str

    report: ModelSelectionReport


def select_best_model(
    candidates: list[ModelSelectionCandidate],
    rules: ModelSelectionRules,
) -> ModelSelectionResult:
    """
    Select the best-performing model from evaluated candidates.

    Supported operations:
    - Input validation
    - Evaluation metric validation
    - Candidate comparison
    - Automatic metric direction
    - Model ranking
    - Best model selection
    - Selection report generation
    """

    # Validate configuration.
    if not isinstance(rules, ModelSelectionRules):
        raise TypeError(
            "Expected ModelSelectionRules configuration."
        )

    if rules.task_type not in (
        "classification",
        "regression",
    ):
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    # Define supported evaluation metrics.
    supported_metrics = {
        "classification": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ],
        "regression": [
            "mae",
            "mse",
            "rmse",
            "r2_score",
        ],
    }

    # Determine the selection metric.
    metric_name = rules.metric_name

    if metric_name is None:
        metric_name = (
            "f1_score"
            if rules.task_type == "classification"
            else "rmse"
        )

    if metric_name not in supported_metrics[rules.task_type]:
        raise ValueError(
            f"Metric '{metric_name}' is not supported "
            f"for {rules.task_type}."
        )

    # Determine metric optimization direction.
    higher_is_better = metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "r2_score",
    )

    # Validate candidates.
    if not isinstance(candidates, list):
        raise TypeError(
            "Expected candidates to be a list."
        )

    if not candidates:
        raise ValueError(
            "At least one model candidate is required."
        )

    candidate_names = []
    scored_candidates = []

    for candidate in candidates:

        if not isinstance(candidate, ModelSelectionCandidate):
            raise TypeError(
                "Every candidate must be a "
                "ModelSelectionCandidate."
            )

        if (
            not isinstance(candidate.model_type, str)
            or not candidate.model_type.strip()
        ):
            raise ValueError(
                "Candidate model_type must be a non-empty string."
            )

        if candidate.model_type in candidate_names:
            raise ValueError(
                f"Duplicate model candidate: "
                f"{candidate.model_type}."
            )

        candidate_names.append(candidate.model_type)

        # Validate model.
        if (
            not hasattr(candidate.model, "predict")
            or not callable(candidate.model.predict)
        ):
            raise TypeError(
                f"Candidate '{candidate.model_type}' "
                "must provide a callable predict method."
            )

        # Validate evaluation report.
        if not isinstance(
            candidate.evaluation_report,
            ModelEvaluationReport,
        ):
            raise TypeError(
                "Expected ModelEvaluationReport "
                "for every candidate."
            )

        if candidate.evaluation_report.task_type != rules.task_type:
            raise ValueError(
                f"Candidate '{candidate.model_type}' "
                "has an incompatible evaluation task type."
            )

        # Extract evaluation score.
        score = getattr(
            candidate.evaluation_report,
            metric_name,
            None,
        )

        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError(
                f"Candidate '{candidate.model_type}' "
                f"has an invalid or missing {metric_name} score."
            )

        score = float(score)

        if not math.isfinite(score):
            raise ValueError(
                f"Candidate '{candidate.model_type}' "
                f"has a non-finite {metric_name} score."
            )

        scored_candidates.append(
            ModelSelectionEntry(
                model_type=candidate.model_type,
                score=score,
            )
        )

    # Rank candidate models.
    rankings = sorted(
        scored_candidates,
        key=lambda entry: entry.score,
        reverse=higher_is_better,
    )

    # Identify the winning model.
    best_model_type = rankings[0].model_type
    best_score = rankings[0].score

    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.model_type == best_model_type
    )

    # Generate selection report.
    report = ModelSelectionReport(
        task_type=rules.task_type,
        metric_name=metric_name,
        selected_model_type=best_model_type,
        selected_score=best_score,
        candidate_count=len(candidates),
        rankings=rankings,
    )

    return ModelSelectionResult(
        selected_model=selected_candidate.model,
        selected_model_type=best_model_type,
        selected_score=best_score,
        metric_name=metric_name,
        report=report,
    )

