
import pandas as pd
import pytest

from evolyrn.data.preprocessing.data_encoder import (
    EncodingRules,
    encode_dataset,
)


def test_one_hot_encoding():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"]
    })

    rules = EncodingRules(
        strategies={"city": "one_hot"}
    )

    result = encode_dataset(df, rules)

    assert "city_Delhi" in result.dataframe.columns
    assert "city_Mumbai" in result.dataframe.columns
    assert "city" not in result.dataframe.columns

    assert result.dataframe["city_Delhi"].tolist() == [1, 0, 1]
    assert result.dataframe["city_Mumbai"].tolist() == [0, 1, 0]


def test_label_encoding():

    df = pd.DataFrame({
        "color": ["Red", "Blue", "Red"]
    })

    rules = EncodingRules(
        strategies={"color": "label"}
    )

    result = encode_dataset(df, rules)

    assert result.dataframe["color"].tolist() == [0, 1, 0]
    assert result.category_mappings["color"] == {
        "Red": 0,
        "Blue": 1,
    }


def test_ordinal_encoding():

    df = pd.DataFrame({
        "level": ["Low", "High", "Medium"]
    })

    rules = EncodingRules(
        strategies={"level": "ordinal"},
        ordinal_categories={
            "level": ["Low", "Medium", "High"]
        }
    )

    result = encode_dataset(df, rules)

    assert result.dataframe["level"].tolist() == [0, 2, 1]


def test_invalid_encoding_strategy():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"]
    })

    rules = EncodingRules(
        strategies={"city": "invalid"}
    )

    with pytest.raises(ValueError):
        encode_dataset(df, rules)


def test_missing_values_rejected():

    df = pd.DataFrame({
        "city": ["Delhi", None, "Mumbai"]
    })

    rules = EncodingRules(
        strategies={"city": "label"}
    )

    with pytest.raises(ValueError, match="missing values"):
        encode_dataset(df, rules)


def test_original_dataframe_not_modified():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"]
    })

    original = df.copy(deep=True)

    rules = EncodingRules(
        strategies={"city": "one_hot"}
    )

    encode_dataset(df, rules)

    pd.testing.assert_frame_equal(df, original)


def test_default_configuration():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"],
        "age": [25, 30]
    })

    result = encode_dataset(df)

    pd.testing.assert_frame_equal(
        result.dataframe,
        df
    )

    assert result.report.strategies_applied == {}


def test_unknown_column_rejected():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"]
    })

    rules = EncodingRules(
        strategies={"country": "label"}
    )

    with pytest.raises(ValueError, match="unknown column"):
        encode_dataset(df, rules)


def test_unknown_ordinal_category_rejected():

    df = pd.DataFrame({
        "level": ["Low", "Expert"]
    })

    rules = EncodingRules(
        strategies={"level": "ordinal"},
        ordinal_categories={
            "level": ["Low", "Medium", "High"]
        }
    )

    with pytest.raises(ValueError, match="Unknown ordinal categories"):
        encode_dataset(df, rules)


def test_generated_column_collision():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"],
        "city_Delhi": [10, 20]
    })

    rules = EncodingRules(
        strategies={"city": "one_hot"}
    )

    with pytest.raises(ValueError, match="already exists"):
        encode_dataset(df, rules)


def test_encoding_report():

    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"]
    })

    rules = EncodingRules(
        strategies={"city": "one_hot"}
    )

    result = encode_dataset(df, rules)

    assert result.report.original_columns == ["city"]

    assert result.report.strategies_applied == {
        "city": "one_hot"
    }

    assert result.report.rows_processed == 3

    assert result.report.encoded_columns["city"] == [
        "city_Delhi",
        "city_Mumbai",
    ]


def test_duplicate_dataframe_columns_rejected():

    df = pd.DataFrame(
        [["Delhi", "Mumbai"]],
        columns=["city", "city"]
    )

    with pytest.raises(ValueError, match="unique column names"):
        encode_dataset(df)
