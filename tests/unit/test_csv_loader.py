import pandas as pd
import pytest

from evolyrn.data.ingestion.csv_loader import load_csv


def test_load_valid_csv(tmp_path):
    """Verify that a valid CSV is loaded correctly."""

    csv_file = tmp_path / "customers.csv"

    csv_file.write_text(
        "age,salary,purchased\n25,30000,0\n32,50000,1\n",
        encoding="utf-8",
    )

    result = load_csv(csv_file)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)
    assert list(result.columns) == ["age", "salary", "purchased"]


def test_missing_file_raises_error(tmp_path):
    """Verify that a missing file raises FileNotFoundError."""

    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing_file)


def test_empty_csv_raises_error(tmp_path):
    """Verify that an empty CSV raises ValueError."""

    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_csv(csv_file)


def test_wrong_file_extension_raises_error(tmp_path):
    """Verify that non-CSV files are rejected."""

    text_file = tmp_path / "customers.txt"
    text_file.write_text("some data", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        load_csv(text_file)


def test_csv_with_headers_only_raises_error(tmp_path):
    """Verify that a CSV without data rows is rejected."""

    csv_file = tmp_path / "headers_only.csv"
    csv_file.write_text(
        "age,salary,purchased\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no data rows"):
        load_csv(csv_file)
