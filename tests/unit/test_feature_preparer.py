
import pandas as pd
import pytest

from evolyrn.preparation.feature_preparer import (
    FeaturePreparationRules,
    prepare_features,
)


def sample_dataframe():
    return pd.DataFrame({
        "Age": [25, 35, 40],
        "Income": [50000, 70000, 90000],
        "Customer ID": ["C101", "C102", "C103"],
        "Purchased": ["Yes", "No", "Yes"],
    })


def test_default_feature_selection():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    result = prepare_features(df, rules)

    assert list(result.X.columns) == [
        "Age", "Income", "Customer ID"
    ]

    assert result.y.tolist() == ["Yes", "No", "Yes"]


def test_exclude_irrelevant_columns():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Customer ID"]
    )

    result = prepare_features(df, rules)

    assert list(result.X.columns) == [
        "Age", "Income"
    ]


def test_explicit_feature_selection():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Income"]
    )

    result = prepare_features(df, rules)

    assert list(result.X.columns) == [
        "Age", "Income"
    ]


def test_feature_selection_order_preserved():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Income", "Age"]
    )

    result = prepare_features(df, rules)

    assert list(result.X.columns) == [
        "Income", "Age"
    ]


def test_missing_target_column_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Revenue"
    )

    with pytest.raises(ValueError, match="does not exist"):
        prepare_features(df, rules)


def test_missing_feature_column_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Salary"]
    )

    with pytest.raises(ValueError, match="does not exist"):
        prepare_features(df, rules)


def test_missing_excluded_column_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Unknown"]
    )

    with pytest.raises(ValueError, match="does not exist"):
        prepare_features(df, rules)


def test_target_column_cannot_be_excluded():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Purchased"]
    )

    with pytest.raises(ValueError, match="cannot be excluded"):
        prepare_features(df, rules)


def test_target_column_cannot_be_feature():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Purchased"]
    )

    with pytest.raises(ValueError, match="cannot be used as a feature"):
        prepare_features(df, rules)


def test_feature_and_exclusion_conflict_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Income"],
        exclude_columns=["Income"]
    )

    with pytest.raises(ValueError, match="also be excluded"):
        prepare_features(df, rules)


def test_empty_feature_selection_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=[]
    )

    with pytest.raises(ValueError, match="At least one feature"):
        prepare_features(df, rules)


def test_no_available_features_rejected():

    df = pd.DataFrame({
        "Purchased": ["Yes", "No"]
    })

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    with pytest.raises(ValueError, match="No feature columns"):
        prepare_features(df, rules)


def test_empty_dataframe_rejected():

    df = pd.DataFrame()

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    with pytest.raises(ValueError, match="empty DataFrame"):
        prepare_features(df, rules)


def test_non_dataframe_input_rejected():

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    with pytest.raises(TypeError, match="Pandas DataFrame"):
        prepare_features(
            ["Yes", "No"],
            rules
        )


def test_invalid_rules_configuration_rejected():

    df = sample_dataframe()

    with pytest.raises(TypeError, match="FeaturePreparationRules"):
        prepare_features(df, None)


def test_duplicate_dataframe_columns_rejected():

    df = pd.DataFrame(
        [[25, 50000, "Yes"]],
        columns=["Age", "Age", "Purchased"]
    )

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    with pytest.raises(ValueError, match="unique column names"):
        prepare_features(df, rules)


def test_duplicate_feature_columns_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Age"]
    )

    with pytest.raises(ValueError, match="Duplicate columns"):
        prepare_features(df, rules)


def test_duplicate_excluded_columns_rejected():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Age", "Age"]
    )

    with pytest.raises(ValueError, match="Duplicate columns"):
        prepare_features(df, rules)


def test_preparation_report():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Customer ID"]
    )

    result = prepare_features(df, rules)

    assert result.report.original_columns == [
        "Age", "Income", "Customer ID", "Purchased"
    ]

    assert result.report.feature_columns == [
        "Age", "Income"
    ]

    assert result.report.target_column == "Purchased"

    assert result.report.excluded_columns == [
        "Customer ID"
    ]

    assert result.report.rows_processed == 3


def test_original_dataframe_not_modified():

    df = sample_dataframe()

    original = df.copy(deep=True)

    rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Customer ID"]
    )

    prepare_features(df, rules)

    pd.testing.assert_frame_equal(df, original)


def test_result_is_independent_of_original_dataframe():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    result = prepare_features(df, rules)

    result.X.loc[0, "Age"] = 999

    assert df.loc[0, "Age"] == 25


def test_feature_and_target_row_alignment():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased"
    )

    result = prepare_features(df, rules)

    assert result.X.index.equals(result.y.index)

    assert len(result.X) == len(result.y)


def test_feature_target_values():

    df = sample_dataframe()

    rules = FeaturePreparationRules(
        target_column="Purchased",
        feature_columns=["Age", "Income"]
    )

    result = prepare_features(df, rules)

    assert result.X["Age"].tolist() == [25, 35, 40]

    assert result.X["Income"].tolist() == [
        50000, 70000, 90000
    ]

    assert result.y.tolist() == [
        "Yes", "No", "Yes"
    ]

