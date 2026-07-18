# ==========================================
# Real-World Data Project - Retail Sales Analysis
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("sales_data.csv")

# Display Dataset
print("First 5 Rows")
print(df.head())

# Dataset Information
print("\nDataset Information")
print(df.info())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Sales Values
if "Sales" in df.columns:
    df["Sales"] = df["Sales"].fillna(df["Sales"].mean())

# Remove Duplicates
df = df.drop_duplicates()

# Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Sales Distribution")
plt.show()

# Sales by Category
if "Category" in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="Category", y="Sales", data=df)
    plt.title("Sales by Category")
    plt.xticks(rotation=45)
    plt.show()

# Sales by Region
if "Region" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Region", y="Sales", data=df)
    plt.title("Sales by Region")
    plt.xticks(rotation=45)
    plt.show()

# Prediction Model
if "Profit" in df.columns and "Sales" in df.columns:

    X = df[["Sales"]]
    y = df["Profit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nModel Performance")
    print("R2 Score:", r2_score(y_test, predictions))
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

    plt.figure(figsize=(8,5))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted Profit")
    plt.show()

# Final Report
print("\n========== PROJECT REPORT ==========")
print("1. Dataset loaded successfully.")
print("2. Missing values handled.")
print("3. Duplicate records removed.")
print("4. Data visualizations created.")
print("5. Correlation analysis completed.")
print("6. Linear Regression model built.")
print("7. Model performance evaluated.")
print("====================================")