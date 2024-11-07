# Liver Cirrhosis Prediction

## Project Overview

This repository contains the code and analysis for predicting liver cirrhosis in patients using various machine learning models. The project involves exploratory data analysis (EDA), data preprocessing, feature engineering, model training, and evaluation using a dataset of clinical parameters.

## Dataset

The dataset includes several features related to liver function:
- **Age**: Patient's age.
- **Gender**: Patient's gender.
- **Total Bilirubin** and **Direct Bilirubin**: Measurements indicating liver function.
- **Total Proteins** and **Albumin**: Indicators of liver and kidney function.
- **A/G Ratio**: The ratio of albumin to globulin.
- **SGPT** and **SGOT**: Enzyme levels related to liver health.
- **Alkaline Phosphatase (Alkphos)**: Enzyme level associated with liver and bone health.

## Problem Statement

Liver cirrhosis is a critical health issue that requires early detection for effective treatment. This project aims to develop predictive models to diagnose liver cirrhosis based on clinical and laboratory data, facilitating earlier intervention and better patient outcomes.

## Project Structure

- `data/`: Contains the dataset used for the project.
- `notebooks/`: Jupyter notebooks with EDA, model training, and evaluation steps.
- `src/`: Python scripts for data preprocessing, model building, and evaluation.
- `README.md`: Project documentation.

## Models and Methodology

The following machine learning models were used for prediction:
- **Logistic Regression**
- **Decision Trees**
- **Random Forests**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Artificial Neural Network (ANN)**

Evaluation metrics include:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Requirements

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- TensorFlow / Keras
- Seaborn and Matplotlib (for visualization)
