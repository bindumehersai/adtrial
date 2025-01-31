import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

data = pd.read_csv("C:/Users/ADMIN/Downloads/Day_16_Healthcare_Data.csv")

print("Initial Data Info:")
print(data.info())
print("\nMissing Values:\n", data.isna().sum())
print("\nDuplicate Rows:", data.duplicated().sum())

num_cols = data.select_dtypes(include=['number']).columns
cat_cols = data.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
data[num_cols] = num_imputer.fit_transform(data[num_cols])

for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.drop_duplicates(inplace=True)

for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

print("\nFinal Data Info:")
print(data.info())
print("\nMissing Values After Cleaning:\n", data.isna().sum())
print("\nDuplicate Rows After Cleaning:", data.duplicated().sum())

data.to_csv("healthcare_cleaned.csv", index=False)
print("\nCleaned dataset saved successfully!")