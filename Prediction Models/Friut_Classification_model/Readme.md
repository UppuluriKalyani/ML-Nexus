# Fruit Classification using PCA and Various Classifiers

## Project Description

This project is centered around the classification of different types of fruit images using Principal Component Analysis (PCA) for dimensionality reduction and applying various machine learning algorithms for classification, such as:

- Support Vector Machine (SVM)
- k-Nearest Neighbors (KNN)
- Decision Tree Classifier

The combination of PCA for feature reduction and classifiers helps streamline the computational complexity while maintaining high classification accuracy.

### Key Technologies Used:
- **Python**: Programming language used for implementation.
- **PCA (Principal Component Analysis)**: Reduces the dimensionality of the dataset to improve performance.
- **SVM**: A powerful algorithm for classification tasks that works by finding the optimal hyperplane to separate classes.
- **KNN**: A simple, intuitive classifier that categorizes data points based on their nearest neighbors.
- **Decision Tree**: A model that splits the dataset based on feature values for classification.

### Libraries and Frameworks:
- **NumPy & Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For implementing PCA, SVM, KNN, and Decision Tree classifiers.
- **Matplotlib & Seaborn**: For visualizing data and results.

## Problem Statement

The project aims to address the challenges associated with high-dimensional data in image classification, such as:

- **High Computational Cost**: Processing large image datasets can be computationally intensive.
- **Feature Redundancy**: High-dimensional data often includes redundant features that do not contribute to model accuracy.
- **Model Overfitting**: Increased complexity can lead to overfitting, where the model performs well on training data but poorly on unseen data.

By applying PCA, this project focuses on retaining the most informative features while reducing dimensionality, thus balancing performance and efficiency. The classifiers are evaluated based on metrics such as accuracy, precision, recall, and F1 score to determine the most effective model for fruit image classification.

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `src/`: Includes scripts for data preprocessing, feature extraction, and training models.
- `notebooks/`: Jupyter notebooks for interactive analysis and visualization.
- `README.md`: Project documentation.
- `results/`: Saved model performance metrics and plots.


