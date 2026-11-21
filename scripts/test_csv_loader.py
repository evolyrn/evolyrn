from pathlib import Path

from evolyrn.data.ingestion.csv_loader import load_csv

# Locate the sample dataset from the project root
dataset_path = Path("examples/sample_customers.csv")

# Load the dataset using EVOLYRN
dataset = load_csv(dataset_path)

print("\nDataset loaded successfully!")

# Display the dataset
print(dataset)

# Display dataset information
print("\nDataset Information:")
dataset.info()

# Display dataset dimensions
print("\nDataset Shape:")
print(dataset.shape)
