from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np
import pickle

# Load data
df = pd.read_csv('data/traffic_data.csv')

# Separate features and target
X = df.drop(columns=['Delay', 'Route'])  # Exclude target variable 'Delay'
y = df['Delay']

# Define categorical and numerical columns
categorical_cols = ['TrafficCondition', 'RoadCondition', 'WeatherCondition', 'TimeOfDay', 'DayType']
numerical_cols = ['BusCapacity', 'AvgSpeed', 'DistanceToDestination']

# Preprocessing pipeline (Scaling and OneHotEncoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_cols)  # Encode categorical features
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Feature Selection using SelectKBest (Choose top k features)
k = 8  # Number of best features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_preprocessed, y)

# Save the preprocessor and selector for future use
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('models/feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
