from evolyrn.data.ingestion.csv_loader import iter_csv_chunks


def main():
    print("\n===== EVOLYRN CHUNK INGESTION TEST =====")

    dataset_path = "data/local/train.csv"
    chunk_size = 200

    total_rows = 0
    total_chunks = 0
    first_records = None
    column_names = None
    data_types = None
    missing_values = None

    # Read the dataset incrementally.
    for chunk in iter_csv_chunks(
        dataset_path,
        chunksize=chunk_size,
    ):
        total_chunks += 1
        total_rows += len(chunk)

        # Capture metadata from the first chunk.
        if first_records is None:
            first_records = chunk.head(5)
            column_names = chunk.columns.tolist()
            data_types = chunk.dtypes

            missing_values = chunk.isnull().sum()

        else:
            missing_values += chunk.isnull().sum()

        print(f"Processed chunk {total_chunks}: {len(chunk)} records")

    print("\n1. Total Chunks:")
    print(total_chunks)

    print("\n2. Total Records Processed:")
    print(total_rows)

    print("\n3. First 5 Records:")
    print(first_records)

    print("\n4. Column Names:")
    print(column_names)

    print("\n5. Data Types:")
    print(data_types)

    print("\n6. Missing Values:")
    print(missing_values)

    print("\n===== CHUNK INGESTION COMPLETED =====")


if __name__ == "__main__":
    main()
