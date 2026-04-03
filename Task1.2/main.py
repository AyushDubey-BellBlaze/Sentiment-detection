import pandas as pd

# Load CSV file
df = pd.read_csv("data.csv")

# Print dataset
print(df)

print("\nFirst 2 rows:")
print(df.head(2))

print("\nColumn names:")
print(df.columns)

print("\nShape of dataset:")
print(df.shape)