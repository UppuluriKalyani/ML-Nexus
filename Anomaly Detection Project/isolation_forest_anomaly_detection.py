# isolation_forest_anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Set Isolation Forest parameters
contamination_rate = 0.05  # Adjust this value based on your dataset and anomaly expectations
max_samples_value = "auto"  # Can be an integer or "auto" (256 or total sample size, whichever is smaller)

# Isolation Forest with custom parameters
iso_forest = IsolationForest(contamination=contamination_rate, max_samples=max_samples_value, random_state=42)
data['Anomaly'] = iso_forest.fit_predict(data)

# Convert labels for anomalies
data['Anomaly'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Save the Isolation Forest model
joblib.dump(iso_forest, 'isolation_forest_model.pkl')
print("Isolation Forest model saved as 'isolation_forest_model.pkl'.")

# Visualize anomalies with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data.drop(columns=['Anomaly']))

plt.figure(figsize=(10, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['Anomaly'], cmap='coolwarm', marker='o', alpha=0.6)
plt.title("Isolation Forest Anomaly Detection with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Anomaly")
plt.savefig('isolation_forest_anomalies_pca.png')
plt.show()
