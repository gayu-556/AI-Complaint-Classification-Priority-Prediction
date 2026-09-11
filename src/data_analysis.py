import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/complaints.csv")

print("========== DATASET INFORMATION ==========\n")

# 1. Display the first 5 rows
print("First 5 rows:")
print(data.head())


# 2. Display the number of rows and columns
print("\nDataset shape:")
print(data.shape)


# 3. Display column names
print("\nColumn names:")
print(data.columns)


# 4. Check missing values
print("\nMissing values:")
print(data.isnull().sum())


# 5. Check duplicate rows
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())


# 6. Display category distribution
print("\nCategory distribution:")
print(data["category"].value_counts())


# 7. Display priority distribution
print("\nPriority distribution:")
print(data["priority"].value_counts())


# 8. Display unique categories
print("\nUnique categories:")
print(data["category"].unique())


# 9. Display unique priorities
print("\nUnique priorities:")
print(data["priority"].unique())

# Remove duplicate rows
data = data.drop_duplicates()

print("\n========== AFTER CLEANING ==========")

# Display the new dataset size
print("Dataset shape after removing duplicates:")
print(data.shape)

# Check duplicates again
print("\nDuplicate rows after cleaning:")
print(data.duplicated().sum())

# Check missing values again
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# Display category distribution after cleaning
print("\nCategory distribution after cleaning:")
print(data["category"].value_counts())

# Display priority distribution after cleaning
print("\nPriority distribution after cleaning:")
print(data["priority"].value_counts())