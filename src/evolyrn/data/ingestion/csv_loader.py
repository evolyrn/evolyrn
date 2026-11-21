from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from evolyrn.security.file_validator import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    validate_csv_file,
)

# Initial resource-control defaults for the MVP.
DEFAULT_MAX_ROWS = 5_000_000
DEFAULT_MAX_COLUMNS = 500
DEFAULT_MAX_FIELD_LENGTH = 100_000


def _validate_dataframe_limits(
    dataframe: pd.DataFrame,
    *,
    max_columns: int,
    max_field_length: int,
) -> None:
    """Validate the structural and field-length limits of a DataFrame."""

    # Check maximum number of columns.
    if len(dataframe.columns) > max_columns:
        raise ValueError("Dataset exceeds the configured column limit.")

    # Check column-name lengths.
    for column in dataframe.columns:
        if len(str(column)) > max_field_length:
            raise ValueError("Column name exceeds the configured field length.")

    # Check string field lengths.
    for column in dataframe.select_dtypes(include=["object", "string"]).columns:
        lengths = dataframe[column].str.len()

        if lengths.max() > max_field_length:
            raise ValueError(
                "Dataset contains a field exceeding the configured maximum length."
            )


def load_csv(
    file_path: str | Path,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
) -> pd.DataFrame:
    """Securely load a CSV dataset with resource limits."""

    # Step 1: Validate configuration.
    limits = {
        "max_rows": max_rows,
        "max_columns": max_columns,
        "max_field_length": max_field_length,
    }

    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    # Step 2: Validate the uploaded file.
    path = validate_csv_file(
        file_path,
        max_file_size_bytes=max_file_size_bytes,
    )

    try:
        # Step 3: Check column count before loading records.
        header = pd.read_csv(path, nrows=0)

        if len(header.columns) > max_columns:
            raise ValueError("Dataset exceeds the configured column limit.")

        # Step 4: Read at most one row beyond the limit.
        dataframe = pd.read_csv(
            path,
            nrows=max_rows + 1,
        )

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file contains no data.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError("The CSV file has an invalid format.") from exc

    except UnicodeDecodeError as exc:
        raise ValueError("The CSV file has an unsupported encoding.") from exc

    # Step 5: Enforce row limits.
    if len(dataframe) > max_rows:
        raise ValueError("Dataset exceeds the configured row limit.")

    if dataframe.empty:
        raise ValueError("The CSV dataset contains no data rows.")

    # Step 6: Enforce field-length limits.
    _validate_dataframe_limits(
        dataframe,
        max_columns=max_columns,
        max_field_length=max_field_length,
    )

    return dataframe


def iter_csv_chunks(
    file_path: str | Path,
    chunksize: int = 100_000,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
) -> Iterator[pd.DataFrame]:
    """Read CSV incrementally with resource limits."""

    settings = {
        "chunksize": chunksize,
        "max_rows": max_rows,
        "max_columns": max_columns,
        "max_field_length": max_field_length,
    }

    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    path = validate_csv_file(
        file_path,
        max_file_size_bytes=max_file_size_bytes,
    )

    reader = None
    total_rows = 0
    has_data = False

    try:
        header = pd.read_csv(path, nrows=0)

        if len(header.columns) > max_columns:
            raise ValueError("Dataset exceeds the configured column limit.")

        reader = pd.read_csv(path, chunksize=chunksize)

        for chunk in reader:
            if chunk.empty:
                continue

            total_rows += len(chunk)

            if total_rows > max_rows:
                raise ValueError("Dataset exceeds the configured row limit.")

            _validate_dataframe_limits(
                chunk,
                max_columns=max_columns,
                max_field_length=max_field_length,
            )

            has_data = True
            yield chunk

        if not has_data:
            raise ValueError("The CSV dataset contains no data rows.")

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file contains no data.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError("The CSV file has an invalid format.") from exc

    except UnicodeDecodeError as exc:
        raise ValueError("The CSV file has an unsupported encoding.") from exc

    finally:
        if reader is not None:
            reader.close()
