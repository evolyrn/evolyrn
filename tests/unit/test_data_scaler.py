
import numpy as np
import pandas as pd
import pytest

from evolyrn.data.preprocessing.data_scaler import (
    ScalingRules,
    scale_dataset,
)


def test_min_max_scaling():

    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    result = scale_dataset(df, rules)

    expected = [0.0, 0.25, 0.5, 0.75, 1.0]

    assert result.dataframe["age"].tolist() == pytest.approx(expected)


def test_standard_scaling():

    df = pd.DataFrame({
        "marks": [10, 20, 30, 40, 50]
    })

    rules = ScalingRules(
        strategies={"marks": "standard"}
    )

    result = scale_dataset(df, rules)

    assert result.dataframe["marks"].mean() == pytest.approx(0.0)
    assert result.dataframe["marks"].std(ddof=0) == pytest.approx(1.0)


def test_custom_min_max_range():

    df = pd.DataFrame({
        "age": [10, 20, 30]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"},
        feature_ranges={"age": (-1.0, 1.0)}
    )

    result = scale_dataset(df, rules)

    assert result.dataframe["age"].tolist() == pytest.approx(
        [-1.0, 0.0, 1.0]
    )


def test_multiple_columns_scaling():

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [1000, 2000, 3000]
    })

    rules = ScalingRules(
        strategies={
            "age": "min_max",
            "income": "standard"
        }
    )

    result = scale_dataset(df, rules)

    assert result.dataframe["age"].min() == pytest.approx(0.0)
    assert result.dataframe["age"].max() == pytest.approx(1.0)

    assert result.dataframe["income"].mean() == pytest.approx(0.0)


def test_original_dataframe_not_modified():

    df = pd.DataFrame({
        "age": [20, 30, 40]
    })

    original = df.copy(deep=True)

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    scale_dataset(df, rules)

    pd.testing.assert_frame_equal(df, original)


def test_default_configuration():

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [1000, 2000, 3000]
    })

    result = scale_dataset(df)

    pd.testing.assert_frame_equal(
        result.dataframe,
        df
    )

    assert result.report.strategies_applied == {}


def test_invalid_strategy():

    df = pd.DataFrame({
        "age": [20, 30, 40]
    })

    rules = ScalingRules(
        strategies={"age": "invalid"}
    )

    with pytest.raises(ValueError, match="Unsupported scaling strategy"):
        scale_dataset(df, rules)


def test_unknown_column_rejected():

    df = pd.DataFrame({
        "age": [20, 30, 40]
    })

    rules = ScalingRules(
        strategies={"salary": "min_max"}
    )

    with pytest.raises(ValueError, match="unknown column"):
        scale_dataset(df, rules)


def test_non_numeric_column_rejected():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Pune"]
    })

    rules = ScalingRules(
        strategies={"city": "min_max"}
    )

    with pytest.raises(TypeError, match="numeric column"):
        scale_dataset(df, rules)


def test_missing_values_rejected():

    df = pd.DataFrame({
        "age": [20, None, 40]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    with pytest.raises(ValueError, match="missing values"):
        scale_dataset(df, rules)


def test_infinite_values_rejected():

    df = pd.DataFrame({
        "age": [20.0, np.inf, 40.0]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    with pytest.raises(ValueError, match="non-finite values"):
        scale_dataset(df, rules)


def test_constant_column_min_max_rejected():

    df = pd.DataFrame({
        "age": [30, 30, 30]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    with pytest.raises(ValueError, match="constant column"):
        scale_dataset(df, rules)


def test_constant_column_standard_rejected():

    df = pd.DataFrame({
        "age": [30, 30, 30]
    })

    rules = ScalingRules(
        strategies={"age": "standard"}
    )

    with pytest.raises(ValueError, match="constant column"):
        scale_dataset(df, rules)


def test_invalid_feature_range_rejected():

    df = pd.DataFrame({
        "age": [10, 20, 30]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"},
        feature_ranges={"age": (5.0, 1.0)}
    )

    with pytest.raises(ValueError, match="must be less"):
        scale_dataset(df, rules)


def test_feature_range_requires_min_max():

    df = pd.DataFrame({
        "age": [10, 20, 30]
    })

    rules = ScalingRules(
        strategies={"age": "standard"},
        feature_ranges={"age": (0.0, 1.0)}
    )

    with pytest.raises(ValueError, match="only be used"):
        scale_dataset(df, rules)


def test_scaling_report():

    df = pd.DataFrame({
        "age": [20, 30, 40]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    result = scale_dataset(df, rules)

    assert result.report.original_columns == ["age"]
    assert result.report.scaled_columns == ["age"]

    assert result.report.strategies_applied == {
        "age": "min_max"
    }

    assert result.report.rows_processed == 3


def test_scaling_parameters_stored():

    df = pd.DataFrame({
        "age": [10, 20, 30]
    })

    rules = ScalingRules(
        strategies={"age": "min_max"}
    )

    result = scale_dataset(df, rules)

    params = result.scaling_parameters["age"]

    assert params["strategy"] == "min_max"
    assert params["min"] == 10.0
    assert params["max"] == 30.0
    assert params["feature_range"] == (0.0, 1.0)


def test_duplicate_dataframe_columns_rejected():

    df = pd.DataFrame(
        [[10, 20]],
        columns=["age", "age"]
    )

    with pytest.raises(ValueError, match="unique column names"):
        scale_dataset(df)

