# New Zealand Migration Analysis

This project analyzes migration patterns in New Zealand using various machine learning techniques.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Data Preparation](#data-preparation)
3. [Feature Engineering](#feature-engineering)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Visualization](#visualization)

## Dependencies

The following Python libraries are required:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install these dependencies using pip:

```
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Data Preparation

1. Load the data from 'migration_nz.csv'.
2. Replace 'Measure' values with numerical representations:
   - 'Arrivals' -> 0
   - 'Departures' -> 1
   - 'Net' -> 2
3. Factorize 'Country' and 'Citizenship' columns.
4. Fill missing 'Value' with median.
5. Drop unnecessary columns ('Country' and 'Citizenship').

## Feature Engineering

1. Apply polynomial features (degree=2) to 'CountryID', 'Measure', 'Year', and 'CitID'.
2. Split the data into training and testing sets (70% train, 30% test).
3. Scale the features using StandardScaler.

## Model Training and Evaluation

Three models are trained and evaluated:

1. Random Forest Regressor
2. Gradient Boosting Regressor
3. XGBoost Regressor

Hyperparameter tuning is performed using GridSearchCV for the Gradient Boosting Regressor.

Models are evaluated using R2 score and RMSE (Root Mean Square Error).

## Visualization

The following visualizations are created:

1. Line plot of migration growth over years
2. Bar plot of migration growth over years
3. Correlation heatmap of the features

To run the analysis, execute the Python script containing the provided code.

Note: Make sure the 'migration_nz.csv' file is in the same directory as the script.
