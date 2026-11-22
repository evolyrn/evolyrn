
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    """Statistical profile of an individual dataset column."""

    name: str
    data_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int

    statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Complete profile of a dataset."""

    row_count: int
    column_count: int
    duplicate_rows: int
    memory_usage_bytes: int

    columns: dict[str, ColumnProfile] = field(default_factory=dict)


def profile_dataset(dataframe: pd.DataFrame) -> DatasetProfile:
    """
    Generate a statistical profile of a Pandas DataFrame.

    This function only analyzes data.
    It does not modify the original DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    row_count = len(dataframe)
    column_count = len(dataframe.columns)

    memory_usage = int(
        dataframe.memory_usage(index=True, deep=True).sum()
    )

    duplicate_rows = int(dataframe.duplicated().sum())

    profile = DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        duplicate_rows=duplicate_rows,
        memory_usage_bytes=memory_usage,
    )

    for column in dataframe.columns:

        series = dataframe[column]

        missing_count = int(series.isna().sum())

        missing_percentage = (
            (missing_count / row_count) * 100
            if row_count > 0
            else 0.0
        )

        unique_count = int(series.nunique(dropna=True))

        statistics = {}

        # Numerical column profiling.
        if pd.api.types.is_numeric_dtype(series):

            statistics = {
                "mean": float(series.mean())
                if series.notna().any() else None,

                "median": float(series.median())
                if series.notna().any() else None,

                "minimum": float(series.min())
                if series.notna().any() else None,

                "maximum": float(series.max())
                if series.notna().any() else None,

                "standard_deviation": float(series.std())
                if series.notna().any() else None,
            }

        # Categorical column profiling.
        else:

            statistics = {
                "top_values": series.value_counts(
                    dropna=True
                ).head(5).to_dict()
            }

        column_profile = ColumnProfile(
            name=str(column),
            data_type=str(series.dtype),
            missing_count=missing_count,
            missing_percentage=round(missing_percentage, 2),
            unique_count=unique_count,
            statistics=statistics,
        )

        profile.columns[str(column)] = column_profile

    return profile
