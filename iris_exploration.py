import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

print("Shape of dataset:", df.shape)
print("\nColumn Names:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary Info:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())


sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

df.hist(figsize=(10, 6), edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

sns.boxplot(data=df.drop("species", axis=1))
plt.title("Boxplot of Features")
plt.show()
