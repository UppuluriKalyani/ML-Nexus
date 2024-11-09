# kmeans_anomaly_detection.py

import pandas as pd
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)
data['Cluster'] = kmeans.labels_

# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans model saved as 'kmeans_model.pkl'.")

# Identify anomalies based on cluster assignment
anomalies_kmeans = data[data['Cluster'] == 1]

# Visualize clusters with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data.drop(columns=['Cluster']))

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Cluster'], cmap='viridis', marker='o', alpha=0.6)
plt.title("KMeans Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.savefig('kmeans_clusters_pca.png')
plt.show()
