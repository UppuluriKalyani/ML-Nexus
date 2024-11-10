# eda_visualizations.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Set plot style
sns.set(style="whitegrid")

# Plot feature distributions
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig('eda_feature_distributions.png')
plt.show()
