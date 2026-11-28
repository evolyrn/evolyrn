
import pandas as pd
import pytest

from evolyrn.data.preprocessing.data_scaler import (
    DataScaler,
    ScalingRules,
)


def test_fit_learns_min_max_parameters():

    train = pd.DataFrame({
        "Age": [20, 30, 40, 50]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    scaler.fit(train)

    state = scaler.get_state()

    assert state["fitted"] is True

    assert state["scaling_parameters"]["Age"]["min"] == 20.0
    assert state["scaling_parameters"]["Age"]["max"] == 50.0


def test_min_max_transform_reuses_training_parameters():

    train = pd.DataFrame({
        "Age": [20, 30, 40, 50]
    })

    validation = pd.DataFrame({
        "Age": [60, 70]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    scaler.fit(train)

    result = scaler.transform(validation)

    assert result.dataframe["Age"].tolist() == pytest.approx(
        [1.3333333333, 1.6666666667]
    )


def test_fit_learns_standard_scaling_parameters():

    train = pd.DataFrame({
        "Salary": [10000, 20000, 30000]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Salary": "standard"}
        )
    )

    scaler.fit(train)

    parameters = (
        scaler.get_state()
        ["scaling_parameters"]["Salary"]
    )

    assert parameters["mean"] == 20000.0
    assert parameters["std"] == pytest.approx(8164.965809)


def test_standard_transform_reuses_training_parameters():

    train = pd.DataFrame({
        "Salary": [10000, 20000, 30000]
    })

    validation = pd.DataFrame({
        "Salary": [40000]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Salary": "standard"}
        )
    )

    scaler.fit(train)

    result = scaler.transform(validation)

    assert result.dataframe["Salary"].iloc[0] == pytest.approx(
        2.449489743
    )


def test_fit_transform_learns_and_transforms():

    train = pd.DataFrame({
        "Age": [20, 30, 40, 50]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    result = scaler.fit_transform(train)

    assert result.dataframe["Age"].tolist() == pytest.approx(
        [0.0, 0.3333333333, 0.6666666667, 1.0]
    )

    assert scaler.get_state()["fitted"] is True


def test_transform_before_fit_rejected():

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    df = pd.DataFrame({
        "Age": [20, 30]
    })

    with pytest.raises(
        RuntimeError,
        match="must be fitted"
    ):
        scaler.transform(df)


def test_original_training_and_validation_unchanged():

    train = pd.DataFrame({
        "Age": [20, 30, 40]
    })

    validation = pd.DataFrame({
        "Age": [50, 60]
    })

    original_train = train.copy(deep=True)
    original_validation = validation.copy(deep=True)

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    scaler.fit_transform(train)
    scaler.transform(validation)

    pd.testing.assert_frame_equal(
        train,
        original_train
    )

    pd.testing.assert_frame_equal(
        validation,
        original_validation
    )


def test_get_state_returns_independent_copy():

    train = pd.DataFrame({
        "Age": [20, 30, 40]
    })

    scaler = DataScaler(
        ScalingRules(
            strategies={"Age": "min_max"}
        )
    )

    scaler.fit(train)

    state = scaler.get_state()

    state["scaling_parameters"]["Age"]["min"] = 999

    assert (
        scaler.get_state()
        ["scaling_parameters"]["Age"]["min"]
        == 20.0
    )

