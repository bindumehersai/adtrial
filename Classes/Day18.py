import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from textblob import TextBlob

data = pd.read_csv("C:/Users/ADMIN/Downloads/Day_18_Tours_and_Travels.csv")

print(data.info())
print(data.describe())
print("Missing values:\n", data.isna().sum())

num_imputer = SimpleImputer(strategy='median')
data['Customer_Age'] = num_imputer.fit_transform(data[['Customer_Age']])

data['Rating'] = data['Rating'].apply(lambda x: np.nan if x not in range(1, 6) else x)
data['Rating'] = num_imputer.fit_transform(data[['Rating']])

data['Review_Text'].fillna("No review provided", inplace=True)

data = data.drop_duplicates(subset=['Review_Text'], keep='first')

data['Rating'] = data['Rating'].clip(1, 5)

def correct_spelling(text):
    return str(TextBlob(text).correct()) if pd.notna(text) else text

data['Destination'] = data['Destination'].apply(correct_spelling)

plt.figure(figsize=(10, 5))
sns.boxplot(x=data['Package_Price'])
plt.title("Boxplot of Package Price (Before Transformation)")
plt.show()

data['Package_Price'] = np.log1p(data['Package_Price'])

label_enc = LabelEncoder()
data['Destination'] = label_enc.fit_transform(data['Destination'])

scaler = StandardScaler()
data[['Customer_Age', 'Package_Price']] = scaler.fit_transform(data[['Customer_Age', 'Package_Price']])

print(data.info())
print("Remaining missing values:\n", data.isna().sum())
print("Duplicates remaining:", data.duplicated().sum())

data.to_csv("cleaned_travel_reviews.csv", index=False)

print("Data cleaning completed successfully!")