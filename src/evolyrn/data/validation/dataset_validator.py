
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationRules:
    """
    Configurable rules for dataset validation.
    """

    required_columns: tuple[str, ...] = ()
    non_nullable_columns: tuple[str, ...] = ()

    numeric_ranges: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )

    check_duplicates: bool = True
    check_empty_columns: bool = True


@dataclass
class ValidationIssue:
    """
    Represents an individual dataset validation finding.
    """

    code: str
    message: str
    severity: str
    column: str | None = None
    count: int = 0


@dataclass
class ValidationReport:
    """
    Contains the complete dataset validation results.
    """

    row_count: int
    column_count: int

    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    empty_columns: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def validate_dataset(
    dataframe: pd.DataFrame,
    rules: ValidationRules | None = None,
) -> ValidationReport:
    """
    Validate a Pandas DataFrame and return a report.

    This function reports data-quality issues.
    It does not modify or reject the original dataset.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if rules is None:
        rules = ValidationRules()

    if not isinstance(rules, ValidationRules):
        raise TypeError("Expected ValidationRules configuration.")

    report = ValidationReport(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )

    if dataframe.empty:
        report.issues.append(
            ValidationIssue(
                code="empty_dataset",
                message="Dataset contains no data rows.",
                severity="error",
            )
        )

    # Check required columns.
    for column in rules.required_columns:
        if column not in dataframe.columns:
            report.issues.append(
                ValidationIssue(
                    code="missing_column",
                    message=f"Required column '{column}' is missing.",
                    severity="error",
                    column=column,
                )
            )

    # Check missing values and empty columns.
    for column in dataframe.columns:

        missing_count = int(dataframe[column].isna().sum())

        if missing_count > 0:
            report.missing_values[column] = missing_count

            severity = (
                "error"
                if column in rules.non_nullable_columns
                else "warning"
            )

            report.issues.append(
                ValidationIssue(
                    code="missing_values",
                    message=f"Column '{column}' contains missing values.",
                    severity=severity,
                    column=column,
                    count=missing_count,
                )
            )

        # Check whether an entire column is empty.
        if (
            rules.check_empty_columns
            and dataframe[column].isna().all()
        ):
            report.empty_columns.append(column)

            report.issues.append(
                ValidationIssue(
                    code="empty_column",
                    message=f"Column '{column}' contains no values.",
                    severity="warning",
                    column=column,
                )
            )

    # Check duplicate records.
    if rules.check_duplicates:
        duplicate_count = int(dataframe.duplicated().sum())

        report.duplicate_rows = duplicate_count

        if duplicate_count > 0:
            report.issues.append(
                ValidationIssue(
                    code="duplicate_rows",
                    message="Dataset contains duplicate records.",
                    severity="warning",
                    count=duplicate_count,
                )
            )

    # Check configurable numeric ranges.
    for column, limits in rules.numeric_ranges.items():

        if column not in dataframe.columns:
            report.issues.append(
                ValidationIssue(
                    code="missing_column",
                    message=f"Numeric rule references missing column '{column}'.",
                    severity="error",
                    column=column,
                )
            )
            continue

        minimum, maximum = limits

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            raise ValueError(
                f"Invalid numeric range configured for '{column}'."
            )

        original = dataframe[column]
        numeric = pd.to_numeric(original, errors="coerce")

        invalid_count = int(
            (original.notna() & numeric.isna()).sum()
        )

        if invalid_count > 0:
            report.issues.append(
                ValidationIssue(
                    code="invalid_numeric_values",
                    message=f"Column '{column}' contains non-numeric values.",
                    severity="error",
                    column=column,
                    count=invalid_count,
                )
            )

        valid_numeric = numeric.dropna()

        out_of_range = pd.Series(False, index=valid_numeric.index)

        if minimum is not None:
            out_of_range |= valid_numeric < minimum

        if maximum is not None:
            out_of_range |= valid_numeric > maximum

        range_violation_count = int(out_of_range.sum())

        if range_violation_count > 0:
            report.issues.append(
                ValidationIssue(
                    code="numeric_range_violation",
                    message=f"Column '{column}' contains values outside the allowed range.",
                    severity="error",
                    column=column,
                    count=range_violation_count,
                )
            )

    return report
