import pytest

from evolyrn.security.file_validator import validate_csv_file


def test_valid_csv_is_accepted(tmp_path):
    csv_file = tmp_path / "customers.csv"
    csv_file.write_text("name,age\nJohn,25\n", encoding="utf-8")

    result = validate_csv_file(csv_file)

    assert result == csv_file


def test_missing_file_is_rejected(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        validate_csv_file(missing_file)


def test_directory_is_rejected(tmp_path):
    directory = tmp_path / "customers.csv"
    directory.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        validate_csv_file(directory)


def test_wrong_extension_is_rejected(tmp_path):
    file = tmp_path / "customers.txt"
    file.write_text("name,age\nJohn,25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        validate_csv_file(file)


def test_empty_file_is_rejected(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        validate_csv_file(file)


def test_oversized_file_is_rejected(tmp_path):
    file = tmp_path / "large.csv"
    file.write_text("name,age\nJohn,25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="size limit"):
        validate_csv_file(file, max_file_size_bytes=5)


def test_custom_size_limit_accepts_file(tmp_path):
    file = tmp_path / "small.csv"
    file.write_text("a,b\n1,2\n", encoding="utf-8")

    result = validate_csv_file(file, max_file_size_bytes=100)

    assert result == file


def test_zero_size_limit_is_rejected(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive"):
        validate_csv_file(file, max_file_size_bytes=0)
