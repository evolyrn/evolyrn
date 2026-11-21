import pandas as pd
import pytest

from evolyrn.data.ingestion.csv_loader import iter_csv_chunks


def test_csv_is_read_in_chunks(tmp_path):
    file = tmp_path / "customers.csv"

    file.write_text(
        "id,name\n1,John\n2,Jane\n3,Alex\n4,Sam\n",
        encoding="utf-8",
    )

    chunks = list(iter_csv_chunks(file, chunksize=2))

    assert len(chunks) == 2
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2


def test_chunk_processing_preserves_all_rows(tmp_path):
    file = tmp_path / "customers.csv"

    file.write_text(
        "id,name\n1,John\n2,Jane\n3,Alex\n",
        encoding="utf-8",
    )

    chunks = list(iter_csv_chunks(file, chunksize=2))

    assert sum(len(chunk) for chunk in chunks) == 3


def test_zero_chunk_size_is_rejected(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text("id,name\n1,John\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive"):
        list(iter_csv_chunks(file, chunksize=0))


def test_negative_chunk_size_is_rejected(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text("id,name\n1,John\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive"):
        list(iter_csv_chunks(file, chunksize=-1))


def test_oversized_file_is_rejected(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text("id,name\n1,John\n", encoding="utf-8")

    with pytest.raises(ValueError, match="size limit"):
        list(
            iter_csv_chunks(
                file,
                chunksize=2,
                max_file_size_bytes=5,
            )
        )


def test_headers_only_csv_is_rejected(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text("id,name\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        list(iter_csv_chunks(file, chunksize=2))


def test_row_limit_is_enforced(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text(
        "id,name\n1,John\n2,Jane\n3,Alex\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row limit"):
        list(
            iter_csv_chunks(
                file,
                chunksize=2,
                max_rows=2,
            )
        )


def test_column_limit_is_enforced(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text(
        "id,name,age\n1,John,25\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="column limit"):
        list(
            iter_csv_chunks(
                file,
                max_columns=2,
            )
        )


def test_field_length_limit_is_enforced(tmp_path):
    file = tmp_path / "customers.csv"
    file.write_text(
        "id,name\n1,Alexander\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field exceeding"):
        list(
            iter_csv_chunks(
                file,
                max_field_length=5,
            )
        )
