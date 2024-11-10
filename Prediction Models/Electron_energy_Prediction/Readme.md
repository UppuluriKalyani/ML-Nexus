# Electron Energy Flux Prediction

## Project Overview

This project aims to develop machine learning models to predict the `ELE_TOTAL_ENERGY_FLUX` of electrons, a measure crucial for space weather analysis and satellite safety. Using a dataset containing telemetry and environmental features, the project evaluates and identifies the optimal model to predict this continuous target variable accurately.

## Problem Statement

The prediction of `ELE_TOTAL_ENERGY_FLUX` is essential in understanding and mitigating the effects of space weather on satellite systems. This project seeks to forecast electron energy flux from a set of features representing environmental conditions and satellite data. This predictive capability aids in satellite operation planning and helps avoid potential damage due to high electron fluxes in the space environment.

## Data

- **Features**: The dataset includes 74 features, including:
  - `SC_AACGM_LAT`: Geographic latitude in the AACGM model.
  - `cos_ut`, `sin_ut`: Sine and cosine transformations of universal time.
  - `cos_doy`, `sin_doy`: Sine and cosine transformations of day of the year.
  - `vsw_10min`, `psw_10min`: Solar wind parameters over a 10-minute interval.
  - And additional telemetry data related to spacecraft operations.

- **Target Variable**: `ELE_TOTAL_ENERGY_FLUX`

## Methods

1. **Data Preprocessing**: The dataset was processed to handle missing values, normalize or scale features, and encode any categorical variables as needed.
2. **Modeling**: Multiple regression models were trained, including:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. **Evaluation**: Models were evaluated based on:
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - Coefficient of Determination (R²)

## Results

The optimal model demonstrated low RMSE and high R², indicating its accuracy and reliability in predicting electron energy flux based on the given features.
