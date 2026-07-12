import pandas as pd

# Load dataset
df = pd.read_csv("sales_data.csv")

# Display first few rows
print(df.head())

# Display column names
print(df.columns)

# Convert Sales to numeric
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

# Check missing values
print(df.isnull().sum())

# Fill missing values
df["Sales"] = df["Sales"].fillna(df["Sales"].mean())

# Remove duplicates
df = df.drop_duplicates()

# Remove outliers using IQR
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Sales"] >= lower) & (df["Sales"] <= upper)]

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned successfully!")