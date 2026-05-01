import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
data = pd.read_csv("data/housing.csv")

# ------------------ HEATMAP ------------------
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.savefig("outputs/heatmap.png")
plt.clf()

# ------------------ HISTOGRAM ------------------
data.hist(figsize=(10,8))
plt.savefig("outputs/histogram.png")
plt.clf()

# ------------------ SCATTER ------------------
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]])
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.savefig("outputs/scatter.png")
plt.clf()

print(data.columns)
print("All images generated successfully!")