
import pandas as pd
import pytest

from evolyrn.data.preprocessing.data_cleaner import (
    CleaningRules,
    clean_dataset,
)


def test_mean_imputation():
    df = pd.DataFrame({
        "age": [20, 30, None, 40]
    })

    rules = CleaningRules(
        missing_value_strategies={"age": "mean"},
        remove_duplicates=False,
    )

    result = clean_dataset(df, rules)

    assert result.dataframe["age"].isna().sum() == 0
    assert result.dataframe.loc[2, "age"] == 30


def test_median_imputation():
    df = pd.DataFrame({
        "salary": [100, 200, None, 400]
    })

    rules = CleaningRules(
        missing_value_strategies={"salary": "median"},
        remove_duplicates=False,
    )

    result = clean_dataset(df, rules)

    assert result.dataframe.loc[2, "salary"] == 200


def test_mode_imputation():
    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", None, "Delhi"]
    })

    rules = CleaningRules(
        missing_value_strategies={"city": "mode"},
        remove_duplicates=False,
    )

    result = clean_dataset(df, rules)

    assert result.dataframe.loc[2, "city"] == "Delhi"


def test_constant_imputation():
    df = pd.DataFrame({
        "city": ["Delhi", None, "Mumbai"]
    })

    rules = CleaningRules(
        missing_value_strategies={"city": "constant"},
        fill_values={"city": "Unknown"},
    )

    result = clean_dataset(df, rules)

    assert result.dataframe.loc[1, "city"] == "Unknown"


def test_drop_rows_with_missing_values():
    df = pd.DataFrame({
        "name": ["John", "Jane", "Alex"],
        "age": [25, None, 30],
    })

    rules = CleaningRules(
        missing_value_strategies={"age": "drop_rows"}
    )

    result = clean_dataset(df, rules)

    assert len(result.dataframe) == 2
    assert result.report.rows_removed_missing == 1


def test_duplicate_removal():
    df = pd.DataFrame({
        "name": ["John", "Jane", "John"],
        "age": [25, 30, 25],
    })

    rules = CleaningRules(
        remove_duplicates=True
    )

    result = clean_dataset(df, rules)

    assert len(result.dataframe) == 2
    assert result.report.duplicates_removed == 1


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({
        "name": ["John", "Jane"],
        "age": [25, None],
    })

    original = df.copy(deep=True)

    rules = CleaningRules(
        missing_value_strategies={"age": "mean"}
    )

    clean_dataset(df, rules)

    pd.testing.assert_frame_equal(df, original)


def test_invalid_strategy_raises_error():
    df = pd.DataFrame({
        "age": [20, None]
    })

    rules = CleaningRules(
        missing_value_strategies={"age": "invalid"}
    )

    with pytest.raises(ValueError):
        clean_dataset(df, rules)


def test_missing_column_rule_raises_error():
    df = pd.DataFrame({
        "age": [20, 30]
    })

    rules = CleaningRules(
        missing_value_strategies={"salary": "mean"}
    )

    with pytest.raises(ValueError):
        clean_dataset(df, rules)


def test_cleaning_report():
    df = pd.DataFrame({
        "age": [20, None, 30]
    })

    rules = CleaningRules(
        missing_value_strategies={"age": "mean"},
        remove_duplicates=False,
    )

    result = clean_dataset(df, rules)

    assert result.report.original_row_count == 3
    assert result.report.final_row_count == 3
    assert result.report.missing_values_before == {"age": 1}
    assert result.report.missing_values_after == {}


# Additional edge-case tests


def test_mean_imputation_all_values_missing():
    df = pd.DataFrame({
        "age": [None, None, None]
    })

    rules = CleaningRules(
        missing_value_strategies={"age": "mean"}
    )

    with pytest.raises(ValueError):
        clean_dataset(df, rules)


def test_median_imputation_all_values_missing():
    df = pd.DataFrame({
        "salary": [None, None, None]
    })

    rules = CleaningRules(
        missing_value_strategies={"salary": "median"}
    )

    with pytest.raises(ValueError):
        clean_dataset(df, rules)


def test_duplicate_keep_last():
    df = pd.DataFrame(
        {
            "name": ["John", "Jane", "John"],
            "age": [25, 30, 25],
        },
        index=[10, 11, 12],
    )

    rules = CleaningRules(
        remove_duplicates=True,
        duplicate_keep="last",
    )

    result = clean_dataset(df, rules)

    assert len(result.dataframe) == 2
    assert list(result.dataframe.index) == [11, 12]
    assert result.dataframe.iloc[-1]["name"] == "John"
    assert result.report.duplicates_removed == 1


def test_invalid_duplicate_keep():
    df = pd.DataFrame({
        "name": ["John", "Jane"]
    })

    rules = CleaningRules(
        duplicate_keep="middle"
    )

    with pytest.raises(ValueError):
        clean_dataset(df, rules)


def test_duplicate_removal_disabled():
    df = pd.DataFrame({
        "name": ["John", "John"],
        "age": [25, 25],
    })

    rules = CleaningRules(
        remove_duplicates=False
    )

    result = clean_dataset(df, rules)

    assert len(result.dataframe) == 2
    assert result.report.duplicates_removed == 0

