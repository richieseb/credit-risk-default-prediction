# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load Dataset
df = pd.read_csv("../data/credit_risk_dataset.csv")
print(f"Dataset Shape: {df.shape}")
display(df.head())

# Cell 3: Data Inspection & Missing Values
print("\n--- Data Info ---")
display(df.info())

print("\n--- Missing Values Percentage ---")
missing = (df.isnull().sum() / len(df)) * 100
display(missing[missing > 0])

print("\n--- Statistical Summary ---")
display(df.describe())

# Cell 4: Target Variable Distribution (Class Imbalance Check)
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df, palette='Set2')
plt.title('Distribution of Loan Status (Default = 1)')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()

default_rate = df['loan_status'].mean() * 100
print(f"Default Rate: {default_rate:.2f}%")

# Cell 5: Numerical Feature Distributions
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols].hist(bins=30, figsize=(15, 12), layout=(4, 3))
plt.tight_layout()
plt.show()

# Cell 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
