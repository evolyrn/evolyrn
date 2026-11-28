
import pandas as pd
import pytest

from evolyrn.data.preprocessing.data_encoder import (
    DataEncoder,
    EncodingRules,
)


def test_fit_learns_category_mappings():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Delhi", "Bangalore"]
    })

    rules = EncodingRules(
        strategies={"City": "label"}
    )

    encoder = DataEncoder(rules)

    encoder.fit(train)

    state = encoder.get_state()

    assert state["fitted"] is True

    assert state["category_mappings"]["City"] == {
        "Delhi": 0,
        "Mumbai": 1,
        "Bangalore": 2,
    }


def test_transform_reuses_training_mappings():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Bangalore"]
    })

    validation = pd.DataFrame({
        "City": ["Mumbai", "Delhi"]
    })

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "label"}
        )
    )

    encoder.fit(train)

    result = encoder.transform(validation)

    assert result.dataframe["City"].tolist() == [1, 0]


def test_unseen_category_rejected_by_default():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai"]
    })

    validation = pd.DataFrame({
        "City": ["Chennai"]
    })

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "label"}
        )
    )

    encoder.fit(train)

    with pytest.raises(
        ValueError,
        match="Unknown categories"
    ):
        encoder.transform(validation)


def test_unseen_category_ignored_when_configured():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai"]
    })

    validation = pd.DataFrame({
        "City": ["Delhi", "Chennai"]
    })

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "label"},
            handle_unknown="ignore",
        )
    )

    encoder.fit(train)

    result = encoder.transform(validation)

    assert result.dataframe["City"].tolist() == [0, -1]


def test_one_hot_transform_reuses_training_columns():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai", "Delhi"]
    })

    validation = pd.DataFrame({
        "City": ["Mumbai", "Delhi"]
    })

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "one_hot"}
        )
    )

    encoder.fit(train)

    result = encoder.transform(validation)

    assert result.dataframe.columns.tolist() == [
        "City_Delhi",
        "City_Mumbai",
    ]

    assert result.dataframe["City_Delhi"].tolist() == [0, 1]

    assert result.dataframe["City_Mumbai"].tolist() == [1, 0]


def test_fit_transform_learns_and_transforms():

    train = pd.DataFrame({
        "Color": ["Red", "Blue", "Green"]
    })

    encoder = DataEncoder(
        EncodingRules(
            strategies={"Color": "label"}
        )
    )

    result = encoder.fit_transform(train)

    assert result.dataframe["Color"].tolist() == [0, 1, 2]

    assert encoder.get_state()["fitted"] is True


def test_transform_before_fit_rejected():

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "label"}
        )
    )

    df = pd.DataFrame({
        "City": ["Delhi"]
    })

    with pytest.raises(
        RuntimeError,
        match="must be fitted"
    ):
        encoder.transform(df)


def test_original_training_and_validation_unchanged():

    train = pd.DataFrame({
        "City": ["Delhi", "Mumbai"]
    })

    validation = pd.DataFrame({
        "City": ["Mumbai", "Delhi"]
    })

    original_train = train.copy(deep=True)
    original_validation = validation.copy(deep=True)

    encoder = DataEncoder(
        EncodingRules(
            strategies={"City": "label"}
        )
    )

    encoder.fit_transform(train)
    encoder.transform(validation)

    pd.testing.assert_frame_equal(
        train,
        original_train
    )

    pd.testing.assert_frame_equal(
        validation,
        original_validation
    )
