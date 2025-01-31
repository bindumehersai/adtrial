import pandas as pd
import numpy as np
import nltk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/ADMIN/Downloads/Day 20_E-Commerce_Data.csv")

print(df.isnull().sum())

numerical_columns = ['Rating', 'Customer_Age']
imputer = SimpleImputer(strategy='median')
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

df['Review_Text'].fillna('No review provided', inplace=True)

print(f"Number of duplicate records: {df.duplicated().sum()}")
df = df.drop_duplicates()

df['Rating'] = df['Rating'].apply(lambda x: min(max(x, 1), 5))
df['Product_Category'] = df['Product_Category'].str.lower()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Product_Price'])
plt.title('Boxplot of Product Price')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Rating'])
plt.title('Boxplot of Rating')
plt.tight_layout()
plt.show()

Q1_price = df['Product_Price'].quantile(0.25)
Q3_price = df['Product_Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

Q1_rating = df['Rating'].quantile(0.25)
Q3_rating = df['Rating'].quantile(0.75)
IQR_rating = Q3_rating - Q1_rating
lower_bound_rating = Q1_rating - 1.5 * IQR_rating
upper_bound_rating = Q3_rating + 1.5 * IQR_rating

df = df[(df['Product_Price'] >= lower_bound_price) & (df['Product_Price'] <= upper_bound_price)]
df = df[(df['Rating'] >= lower_bound_rating) & (df['Rating'] <= upper_bound_rating)]

label_encoder = LabelEncoder()
df['Product_Category'] = label_encoder.fit_transform(df['Product_Category'])

df.to_csv('cleaned_ecommerce_reviews.csv', index=False)

print(df.head())