# ==========================================
# Exploratory Data Analysis (EDA) Project
# Dataset: Titanic
# ==========================================

# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset File
file_name = "titanic.csv"

# Check if file exists
if not os.path.exists(file_name):
    print("Error: titanic.csv file not found!")
    exit()

# Load Dataset
try:
    df = pd.read_csv(file_name)

    if df.empty:
        print("Error: Dataset is empty.")
        exit()

except pd.errors.EmptyDataError:
    print("Error: titanic.csv is empty or not a valid CSV file.")
    exit()

except Exception as e:
    print("Error while reading dataset:", e)
    exit()

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin Column
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Remove Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Survival Count
if "Survived" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count")
    plt.show()

# Passenger Class
if "Pclass" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Pclass", data=df)
    plt.title("Passenger Class Distribution")
    plt.show()

# Gender
if "Sex" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Sex", data=df)
    plt.title("Gender Distribution")
    plt.show()

# Age Distribution
if "Age" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["Age"], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.show()

# Survival by Gender
if "Sex" in df.columns and "Survived" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival by Gender")
    plt.show()

# Survival by Passenger Class
if "Pclass" in df.columns and "Survived" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="Pclass", hue="Survived", data=df)
    plt.title("Survival by Passenger Class")
    plt.show()

# Fare Distribution
if "Fare" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["Fare"])
    plt.title("Fare Distribution")
    plt.show()

# Pair Plot
required = ["Age", "Fare", "Pclass", "Survived"]

if all(col in df.columns for col in required):
    sns.pairplot(df[required], hue="Survived")
    plt.show()

# Final Report
print("\n========== EDA REPORT ==========")
print("Dataset loaded successfully.")
print("Missing values handled.")
print("Duplicate rows removed.")
print("Visualizations generated.")
print("Correlation analysis completed.")
print("================================")