# Water Potability Analysis System

## Project Overview

The **Water Potability Analysis System** is designed to classify whether water is potable (safe for drinking) based on various physicochemical properties. The system leverages machine learning models to predict the potability of water samples using factors like pH, hardness, solids, chloramines, and more.

## Motivation

Access to clean drinking water is vital for maintaining health. According to studies, contaminated water sources can lead to severe health complications. Therefore, this project aims to automate the process of determining water quality using data-driven techniques.

## Dataset Information

The dataset used in this project contains information on the following features:

- **pH**: Measure of acidity or basicity.
- **Hardness**: Measures the concentration of calcium and magnesium.
- **Solids**: Total dissolved solids in ppm.
- **Chloramines**: Amount of chloramines in ppm.
- **Sulfate**: Amount of sulfate in ppm.
- **Conductivity**: Electrical conductivity in μS/cm.
- **Organic Carbon**: Organic matter concentration in ppm.
- **Trihalomethanes**: Concentration of trihalomethanes in ppm.
- **Turbidity**: Measure of water clarity.
- **Potability**: Target variable indicating if water is safe to drink.

The dataset contains 3,276 entries with some missing values in features like pH and sulfate.

## Project Workflow

1. **Data Preprocessing**:
   - Handling missing values using imputation techniques.
   - Feature scaling and normalization.
2. **Exploratory Data Analysis (EDA)**:
   - Analyzing data distribution, correlations, and visualizing patterns.
3. **Data Balancing**:
   - Addressing class imbalance using SMOTE.
4. **Model Building**:
   - Training multiple models including RandomForest, XGBoost, etc.
   - Tuning hyperparameters for optimized performance.
5. **Model Evaluation**:
   - Evaluating model performance using metrics like ROC-AUC, F1-score, etc.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `imbalanced-learn`, `xgboost`
- **Jupyter Notebook**: For code execution and visualization

## Model Evaluation

The models were evaluated using multiple metrics including:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives.
- **F1-Score**: The weighted average of Precision and Recall.
- **ROC-AUC**: The area under the Receiver Operating Characteristic curve, indicating the model’s ability to distinguish between classes.

### Best Model Performance

The **RandomForest Classifier** performed best among the evaluated models, achieving:
**ROC-AUC Score**: 75%
