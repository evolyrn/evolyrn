
import pandas as pd

from evolyrn.data.preprocessing.data_cleaner import (
    DataCleaner,
    CleaningRules,
)


# 1. Create training dataset.
train = pd.DataFrame({
    "Age": [20, 25, 30, 35, None],
    "Salary": [20000, 30000, 40000, 50000, 60000],
})

# 2. Create validation dataset.
validation = pd.DataFrame({
    "Age": [None, 40],
    "Salary": [None, 70000],
})

# Preserve original datasets for verification.
original_train = train.copy(deep=True)
original_validation = validation.copy(deep=True)

# 3. Configure cleaning rules.
rules = CleaningRules(
    missing_value_strategies={
        "Age": "median",
        "Salary": "mean",
    },
)

# 4. Initialize reusable cleaner.
cleaner = DataCleaner(rules)

# 5. Learn parameters and clean training data.
train_result = cleaner.fit_transform(train)

# 6. Transform validation data using learned parameters.
validation_result = cleaner.transform(validation)

# 7. Display learned state.
print("\n========== LEARNED CLEANING STATE ==========")
print(cleaner.get_state())

# 8. Display cleaned training data.
print("\n========== CLEANED TRAINING DATA ==========")
print(train_result.dataframe)

# 9. Display cleaned validation data.
print("\n========== CLEANED VALIDATION DATA ==========")
print(validation_result.dataframe)

# 10. Verify learned parameters.
state = cleaner.get_state()["learned_parameters"]

assert state["Age"]["value"] == 27.5
assert state["Salary"]["value"] == 40000.0

# 11. Verify validation uses training parameters.
assert validation_result.dataframe.loc[0, "Age"] == 27.5
assert validation_result.dataframe.loc[0, "Salary"] == 40000.0

# 12. Verify original datasets are unchanged.
pd.testing.assert_frame_equal(train, original_train)
pd.testing.assert_frame_equal(validation, original_validation)

# 13. Verify no missing values remain.
assert not train_result.dataframe.isnull().any().any()
assert not validation_result.dataframe.isnull().any().any()

print("\nALL DATA CLEANER LIFECYCLE CHECKS PASSED!")


# ============================================================
# ADDITIONAL TEST: CATEGORICAL DATA CLEANING
# ============================================================

print("\n========== CATEGORICAL CLEANING TEST ==========")

train_categorical = pd.DataFrame({
    "City": ["Delhi", "Mumbai", "Delhi", None, "Delhi"],
})

validation_categorical = pd.DataFrame({
    "City": ["Chennai", None, "Mumbai"],
})

categorical_rules = CleaningRules(
    missing_value_strategies={
        "City": "mode",
    },
    remove_duplicates=False,
)

categorical_cleaner = DataCleaner(categorical_rules)

# Learn mode from training data.
categorical_train_result = (
    categorical_cleaner.fit_transform(train_categorical)
)

# Reuse learned mode on validation data.
categorical_validation_result = (
    categorical_cleaner.transform(validation_categorical)
)

print("Learned state:")
print(categorical_cleaner.get_state())

print("\nCleaned validation data:")
print(categorical_validation_result.dataframe)

# Verify learned mode.
assert (
    categorical_cleaner.get_state()
    ["learned_parameters"]["City"]["value"]
    == "Delhi"
)

# Verify missing category is replaced using training mode.
assert (
    categorical_validation_result.dataframe.loc[1, "City"]
    == "Delhi"
)

# Verify previously unseen category remains unchanged.
assert (
    categorical_validation_result.dataframe.loc[0, "City"]
    == "Chennai"
)

print("\nCATEGORICAL LIFECYCLE TEST PASSED!")