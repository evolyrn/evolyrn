
import pandas as pd
import pytest

from evolyrn.data.validation.dataset_validator import (
    ValidationRules,
    validate_dataset,
)


def test_valid_dataset_returns_clean_report():

    df = pd.DataFrame({
        "name": ["John", "Jane"],
        "age": [25, 30],
        "salary": [30000, 50000],
    })

    report = validate_dataset(df)

    assert report.row_count == 2
    assert report.column_count == 3
    assert report.duplicate_rows == 0
    assert report.has_errors is False


def test_missing_values_are_reported():

    df = pd.DataFrame({
        "name": ["John", None],
        "age": [25, 30],
    })

    report = validate_dataset(df)

    assert report.missing_values["name"] == 1


def test_non_nullable_column_generates_error():

    df = pd.DataFrame({
        "name": ["John", None],
    })

    rules = ValidationRules(
        non_nullable_columns=("name",)
    )

    report = validate_dataset(df, rules)

    assert report.has_errors is True


def test_duplicate_rows_are_detected():

    df = pd.DataFrame({
        "name": ["John", "John"],
        "age": [25, 25],
    })

    report = validate_dataset(df)

    assert report.duplicate_rows == 1


def test_empty_columns_are_detected():

    df = pd.DataFrame({
        "name": ["John", "Jane"],
        "age": [None, None],
    })

    report = validate_dataset(df)

    assert "age" in report.empty_columns


def test_required_column_is_checked():

    df = pd.DataFrame({
        "name": ["John"],
    })

    rules = ValidationRules(
        required_columns=("name", "age")
    )

    report = validate_dataset(df, rules)

    assert report.has_errors is True


def test_numeric_range_violation_is_detected():

    df = pd.DataFrame({
        "age": [25, 150, 30],
    })

    rules = ValidationRules(
        numeric_ranges={"age": (18, 100)}
    )

    report = validate_dataset(df, rules)

    issue = next(
        issue for issue in report.issues
        if issue.code == "numeric_range_violation"
    )

    assert issue.count == 1


def test_non_numeric_values_are_detected():

    df = pd.DataFrame({
        "age": [25, "unknown", 30],
    })

    rules = ValidationRules(
        numeric_ranges={"age": (18, 100)}
    )

    report = validate_dataset(df, rules)

    assert any(
        issue.code == "invalid_numeric_values"
        for issue in report.issues
    )


def test_original_dataframe_is_not_modified():

    df = pd.DataFrame({
        "name": ["John", None],
    })

    original = df.copy(deep=True)

    validate_dataset(df)

    pd.testing.assert_frame_equal(df, original)


def test_non_dataframe_input_is_rejected():

    with pytest.raises(TypeError):
        validate_dataset(["John", "Jane"])
