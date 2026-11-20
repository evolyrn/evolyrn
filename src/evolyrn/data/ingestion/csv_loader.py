from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset into a Pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A Pandas DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or invalid.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {path}")

    try:
        dataframe = pd.read_csv(path)

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc

    except pd.errors.ParserError as exc:
        raise ValueError("The CSV file has an invalid format.") from exc

    if dataframe.empty:
        raise ValueError("The CSV dataset contains no data rows.")

    return dataframe
