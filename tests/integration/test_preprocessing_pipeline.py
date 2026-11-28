
import pandas as pd
import pytest

from evolyrn.preparation.preprocessing_pipeline import (
    UnifiedPreprocessingPipeline,
)

from evolyrn.preparation.feature_preparer import (
    FeaturePreparationRules,
)

from evolyrn.data.preprocessing.data_cleaner import (
    CleaningRules,
)

from evolyrn.data.preprocessing.data_encoder import (
    EncodingRules,
)

from evolyrn.data.preprocessing.data_scaler import (
    ScalingRules,
)


# ============================================================
# SAMPLE DATASETS
# ============================================================

def training_dataset():

    return pd.DataFrame({
        "Age": [20, 25, None, 35, 40],
        "Salary": [20000, None, 40000, 50000, 60000],
        "City": ["Delhi", "Mumbai", "Delhi", None, "Mumbai"],
        "Customer_ID": ["C101", "C102", "C103", "C104", "C105"],
        "Purchased": ["Yes", "No", "Yes", "No", "Yes"],
    })


def validation_dataset():

    return pd.DataFrame({
        "Age": [None, 45],
        "Salary": [None, 70000],
        "City": ["Delhi", "Mumbai"],
        "Customer_ID": ["C201", "C202"],
        "Purchased": ["No", "Yes"],
    })


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

def create_pipeline():

    feature_rules = FeaturePreparationRules(
        target_column="Purchased",
        exclude_columns=["Customer_ID"],
    )

    cleaning_rules = CleaningRules(
        missing_value_strategies={
            "Age": "median",
            "Salary": "mean",
            "City": "mode",
        },
        remove_duplicates=True,
    )

    encoding_rules = EncodingRules(
        strategies={
            "City": "one_hot",
        },
    )

    scaling_rules = ScalingRules(
        strategies={
            "Age": "min_max",
            "Salary": "standard",
        },
    )

    return UnifiedPreprocessingPipeline(
        feature_rules=feature_rules,
        cleaning_rules=cleaning_rules,
        encoding_rules=encoding_rules,
        scaling_rules=scaling_rules,
    )


# ============================================================
# TEST 1: COMPLETE TRAINING WORKFLOW
# ============================================================

def test_complete_training_workflow():

    train = training_dataset()

    pipeline = create_pipeline()

    result = pipeline.fit_transform(train)

    assert not result.X.empty

    assert len(result.X) == len(result.y)

    assert "Age" in result.X.columns

    assert "Salary" in result.X.columns

    assert "City_Delhi" in result.X.columns

    assert "City_Mumbai" in result.X.columns

    assert "Customer_ID" not in result.X.columns

    assert result.y.tolist() == [
        "Yes", "No", "Yes", "No", "Yes"
    ]


# ============================================================
# TEST 2: VALIDATION USES TRAINING PARAMETERS
# ============================================================

def test_validation_reuses_training_parameters():

    train = training_dataset()

    validation = validation_dataset()

    pipeline = create_pipeline()

    pipeline.fit(train)

    result = pipeline.transform(validation)

    state = pipeline.get_state()

    # Age median learned from training:
    # [20, 25, 30, 35, 40] -> 30

    assert state["cleaning_state"][
        "learned_parameters"
    ]["Age"]["value"] == 30.0

    # Salary mean learned from training:
    # (20000 + 40000 + 50000 + 60000) / 4

    assert state["cleaning_state"][
        "learned_parameters"
    ]["Salary"]["value"] == 42500.0

    assert len(result.X) == 2

    assert len(result.y) == 2


# ============================================================
# TEST 3: TARGET IS NOT ENCODED OR SCALED
# ============================================================

def test_target_remains_unchanged():

    train = training_dataset()

    pipeline = create_pipeline()

    result = pipeline.fit_transform(train)

    assert result.y.tolist() == train["Purchased"].tolist()

    assert "Purchased" not in result.X.columns


# ============================================================
# TEST 4: ORIGINAL DATASET IS NOT MODIFIED
# ============================================================

def test_original_training_dataset_unchanged():

    train = training_dataset()

    original = train.copy(deep=True)

    pipeline = create_pipeline()

    pipeline.fit_transform(train)

    pd.testing.assert_frame_equal(
        train,
        original,
    )


# ============================================================
# TEST 5: VALIDATION DOES NOT CHANGE LEARNED STATE
# ============================================================

def test_validation_does_not_modify_pipeline_state():

    pipeline = create_pipeline()

    pipeline.fit(training_dataset())

    original_state = pipeline.get_state()

    pipeline.transform(validation_dataset())

    updated_state = pipeline.get_state()

    assert original_state == updated_state


# ============================================================
# TEST 6: FEATURE AND TARGET ROW ALIGNMENT
# ============================================================

def test_feature_target_alignment():

    pipeline = create_pipeline()

    result = pipeline.fit_transform(training_dataset())

    assert result.X.index.equals(result.y.index)

    assert len(result.X) == len(result.y)


# ============================================================
# TEST 7: PIPELINE STATE CONTAINS ALL COMPONENTS
# ============================================================

def test_pipeline_state_contains_all_components():

    pipeline = create_pipeline()

    pipeline.fit(training_dataset())

    state = pipeline.get_state()

    assert state["fitted"] is True

    assert "cleaning_state" in state

    assert "encoding_state" in state

    assert "scaling_state" in state

    assert state["target_column"] == "Purchased"


# ============================================================
# TEST 8: TRANSFORM BEFORE FIT IS REJECTED
# ============================================================

def test_transform_before_fit_rejected():

    pipeline = create_pipeline()

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        pipeline.transform(validation_dataset())


# ============================================================
# TEST 9: INVALID INPUT REJECTED
# ============================================================

def test_non_dataframe_input_rejected():

    pipeline = create_pipeline()

    with pytest.raises(
        TypeError,
        match="Pandas DataFrame",
    ):
        pipeline.fit(["invalid", "input"])


# ============================================================
# TEST 10: TRAINING AND VALIDATION SCHEMA MISMATCH
# ============================================================

def test_validation_schema_mismatch_rejected():

    pipeline = create_pipeline()

    pipeline.fit(training_dataset())

    invalid_validation = validation_dataset().drop(
        columns=["Age"]
    )

    with pytest.raises(
        ValueError,
        match="columns must match",
    ):
        pipeline.transform(invalid_validation)


# ============================================================
# TEST 11: EMPTY DATASET REJECTED
# ============================================================

def test_empty_dataframe_rejected():

    pipeline = create_pipeline()

    with pytest.raises(
        ValueError,
        match="empty DataFrame",
    ):
        pipeline.fit(pd.DataFrame())


# ============================================================
# TEST 12: DUPLICATE ROW REMOVAL PRESERVES ALIGNMENT
# ============================================================

def test_duplicate_removal_preserves_alignment():

    train = pd.DataFrame({
        "Age": [20, 20, 30],
        "Salary": [20000, 20000, 30000],
        "City": ["Delhi", "Delhi", "Mumbai"],
        "Purchased": ["Yes", "Yes", "No"],
    })

    pipeline = UnifiedPreprocessingPipeline(
        feature_rules=FeaturePreparationRules(
            target_column="Purchased",
        ),
        cleaning_rules=CleaningRules(
            missing_value_strategies={},
            remove_duplicates=True,
        ),
        encoding_rules=EncodingRules(
            strategies={"City": "one_hot"},
        ),
        scaling_rules=ScalingRules(
            strategies={"Age": "min_max"},
        ),
    )

    result = pipeline.fit_transform(train)

    assert len(result.X) == 2

    assert len(result.y) == 2

    assert result.X.index.equals(result.y.index)

    assert result.y.tolist() == ["Yes", "No"]
