# anomalies_report.py

import pandas as pd

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Identify anomalies based on both clustering and isolation forest
anomalies_kmeans = data[data['Cluster'] == 1]  # From KMeans
anomalies_isoforest = data[data['Anomaly'] == 1]  # From Isolation Forest

# Combine and save identified anomalies from both methods
anomalies_report = pd.concat([anomalies_kmeans, anomalies_isoforest]).drop_duplicates()
anomalies_report.to_csv('anomalies_report.csv', index=False)
print("Anomalies report saved as 'anomalies_report.csv'.")
