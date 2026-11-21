from pathlib import Path

# Initial local-development limit.
# This will eventually be configurable through deployment settings.
DEFAULT_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


def validate_csv_file(
    file_path: str | Path,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Path:
    """
    Perform basic security checks on a CSV file before ingestion.

    Checks:
        1. File path is valid.
        2. File exists.
        3. File is not a symbolic link.
        4. File is a regular file.
        5. File extension is CSV.
        6. File size is within the configured limit.

    Returns:
        Validated file path.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails validation.
    """

    path = Path(file_path)

    if max_file_size_bytes <= 0:
        raise ValueError("Maximum file size must be positive.")

    if path.is_symlink():
        raise ValueError("Symbolic links are not accepted.")

    if not path.exists():
        raise FileNotFoundError("Dataset not found.")

    if not path.is_file():
        raise ValueError("Expected a regular file.")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    file_size = path.stat().st_size

    if file_size == 0:
        raise ValueError("The CSV file is empty.")

    if file_size > max_file_size_bytes:
        raise ValueError("CSV file exceeds the configured size limit.")

    return path
