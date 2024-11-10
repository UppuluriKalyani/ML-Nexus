# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('raw_data.csv')  # Replace with the actual path to your dataset

# Check for missing values and drop rows with missing values
data = data.dropna()

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Save preprocessed data to CSV
data_scaled.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'preprocessed_data.csv'.")
