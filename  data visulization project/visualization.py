import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_data.csv")

# Bar Chart
plt.figure(figsize=(6,4))
sns.barplot(x="City", y="Sales", data=df)
plt.title("Sales by City")
plt.show()

# Pie Chart
city_sales = df.groupby("City")["Sales"].sum()

plt.figure(figsize=(6,6))
plt.pie(city_sales,
        labels=city_sales.index,
        autopct='%1.1f%%')
plt.title("Sales Distribution")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["Sales"], bins=10)
plt.title("Sales Distribution")
plt.show()

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()