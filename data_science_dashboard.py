import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load sample dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Data Science Dashboard")
print("=" * 50)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Create visualizations
plt.figure(figsize=(12, 8))

# Scatter plot
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Sepal Length vs Width')

# Box plot
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='species', y='petal length (cm)')
plt.title('Petal Length by Species')

# Histogram
plt.subplot(2, 2, 3)
df['sepal length (cm)'].hist(bins=20)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')

# Pair plot (simplified)
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Petal Length vs Width')

plt.tight_layout()
plt.savefig('iris_dashboard.png')
print("\nDashboard visualization saved as 'iris_dashboard.png'")

# Group by species and calculate means
print("\nMean values by species:")
print(df.groupby('species').mean())
