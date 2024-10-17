# ARIMA Model Development for Time Series Forecasting

## Overview

This notebook contains the code and resources for developing an ARIMA model to forecast time series data. The model uses historical data to make predictions on future values, specifically focusing on stock prices.

## Table of Contents

- [Database Changes](#database-changes)
- [ARIMA Model Development Process](#arima-model-development-process)
- [Usage](#usage)

## Database Changes

To facilitate the development of the ARIMA model, the following changes were made to the database:

1. **New Features Added**:

   - **Day of Week**: Extracted from the date to capture weekly seasonality.
   - **Month**: Extracted from the date to capture monthly trends.
   - **Lagged Close Price**: Included the previous day's closing price as a feature for better prediction accuracy.
   - **Moving Averages**: Added 3-day and 7-day moving averages to smooth the data and identify trends.
   - **Volume**: Included volume as a feature to understand its impact on stock prices.

2. **Data Preprocessing**:
   - Rows with missing values were removed after feature engineering to ensure model accuracy.
   - Features were standardized using `StandardScaler` to improve model performance.

## ARIMA Model Development Process

The following steps outline the process for developing the ARIMA model:

1. **Data Loading**:

   - Load historical stock price data from a CSV file, parsing dates correctly.

2. **Feature Engineering**:

   - Extract features like day of the week, month, lagged prices, moving averages, and volume.
   - Drop any rows with NaN values resulting from feature engineering.

3. **Data Splitting**:

   - Split the dataset into training and validation sets (80% training, 20% validation).

4. **Model Specification**:

   - Specify the order of the ARIMA model (p, d, q) based on prior analysis or domain knowledge.
   - Define seasonal orders if applicable.

5. **Model Fitting**:

   - Fit the SARIMAX model using the training dataset, including exogenous features.

6. **Model Saving**:

   - Save the trained model and scaler using `pickle` for later use in forecasting.

7. **Model Testing**:

   - Load the saved model and scaler.
   - Preprocess the test dataset in the same way as the training dataset.
   - Make predictions using the test dataset and evaluate model performance using RMSE (Root Mean Square Error).

8. **Residual Analysis**:
   - Analyze the residuals of the model to check for any patterns that may indicate model inadequacy.

## Usage

1. **Install Dependencies**:
   Make sure you have the required packages installed. You can do this using `pip`:

   ```bash
   pip install pandas numpy scikit-learn statsmodels
   ```
