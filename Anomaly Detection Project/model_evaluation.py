# model_evaluation.py

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load data with true labels (replace 'TrueLabels' with the actual column if available)
data = pd.read_csv('preprocessed_data.csv')
# Assuming `TrueLabels` column exists in the original data

# Evaluate Isolation Forest Model
# Assuming 'TrueLabels' is in the original data and represents ground truth for anomalies
# Uncomment and use if true labels are available

# true_labels = data['TrueLabels']
# print("Confusion Matrix (Isolation Forest):\n", confusion_matrix(true_labels, data['Anomaly']))
# print("Classification Report (Isolation Forest):\n", classification_report(true_labels, data['Anomaly']))

# Save the evaluation report to file
# with open("evaluation_report.txt", "w") as f:
#     f.write("Confusion Matrix (Isolation Forest):\n")
#     f.write(str(confusion_matrix(true_labels, data['Anomaly'])) + "\n\n")
#     f.write("Classification Report (Isolation Forest):\n")
#     f.write(classification_report(true_labels, data['Anomaly']))
