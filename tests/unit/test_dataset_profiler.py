
import pandas as pd
import pytest

from evolyrn.data.profiling.dataset_profiler import profile_dataset


def test_dataset_overview():

    df = pd.DataFrame({
        "name": ["John", "Jane"],
        "age": [25, 30],
    })

    profile = profile_dataset(df)

    assert profile.row_count == 2
    assert profile.column_count == 2


def test_numerical_statistics():

    df = pd.DataFrame({
        "age": [20, 30, 40],
    })

    profile = profile_dataset(df)

    age = profile.columns["age"]

    assert age.statistics["mean"] == 30
    assert age.statistics["median"] == 30
    assert age.statistics["minimum"] == 20
    assert age.statistics["maximum"] == 40


def test_missing_values_are_profiled():

    df = pd.DataFrame({
        "age": [20, None, 40, None],
    })

    profile = profile_dataset(df)

    age = profile.columns["age"]

    assert age.missing_count == 2
    assert age.missing_percentage == 50.0


def test_unique_values_are_counted():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    profile = profile_dataset(df)

    assert profile.columns["city"].unique_count == 2


def test_categorical_top_values():

    df = pd.DataFrame({
        "city": ["Delhi", "Delhi", "Mumbai"],
    })

    profile = profile_dataset(df)

    top_values = profile.columns["city"].statistics["top_values"]

    assert top_values["Delhi"] == 2


def test_duplicate_rows_are_detected():

    df = pd.DataFrame({
        "name": ["John", "John", "Jane"],
        "age": [25, 25, 30],
    })

    profile = profile_dataset(df)

    assert profile.duplicate_rows == 1


def test_memory_usage_is_reported():

    df = pd.DataFrame({
        "age": [20, 30, 40],
    })

    profile = profile_dataset(df)

    assert profile.memory_usage_bytes > 0


def test_empty_dataframe_is_supported():

    df = pd.DataFrame(columns=["name", "age"])

    profile = profile_dataset(df)

    assert profile.row_count == 0
    assert profile.column_count == 2


def test_non_dataframe_input_is_rejected():

    with pytest.raises(TypeError):
        profile_dataset(["John", "Jane"])


def test_original_dataframe_is_not_modified():

    df = pd.DataFrame({
        "name": ["John", "Jane"],
        "age": [25, 30],
    })

    original = df.copy(deep=True)

    profile_dataset(df)

    pd.testing.assert_frame_equal(df, original)
