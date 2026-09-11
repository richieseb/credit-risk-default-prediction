# Cell 1: Imports & Data Loading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/credit_risk_dataset.csv")

# Cell 2: Missing Value Imputation Strategy
# Impute numerical columns with median, categorical with mode
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after imputation:", df.isnull().sum().sum())

# Cell 3: Feature Engineering
# Create Loan-to-Income ratio
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

# Create Interest-to-Income ratio proxy if applicable
if 'loan_int_rate' in df.columns:
    df['interest_burden'] = df['loan_int_rate'] * df['loan_amnt'] / df['person_income']

print(df[['loan_amnt', 'person_income', 'loan_to_income_ratio']].head())

# Cell 4: Categorical Encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"Encoded shape: {df_encoded.shape}")

# Cell 5: Train-Test Split & SMOTE for Class Imbalance
X = df_encoded.drop(columns=['loan_status'])
y = df_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Before SMOTE - Training shape: {X_train.shape}, Defaults: {y_train.sum()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training shape: {X_train_resampled.shape}, Defaults: {y_train_resampled.sum()}")