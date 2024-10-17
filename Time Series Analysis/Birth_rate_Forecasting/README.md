# Time Series Forecasting: Daily Female Births

## Project Overview

This project focuses on time series forecasting using various models to predict daily female births. The dataset used is "daily-total-female-births.csv", which contains information about daily female births in 1959.

## Models Implemented

1. Prophet
2. ARIMA (AutoRegressive Integrated Moving Average)
3. LSTM (Long Short-Term Memory)

## Requirements

To run this project, you'll need the following Python libraries:

- pandas
- numpy
- prophet
- statsmodels
- matplotlib
- seaborn
- tensorflow
- scikit-learn

You can install these libraries using pip:

```
pip install pandas numpy prophet statsmodels matplotlib seaborn tensorflow scikit-learn
```

## File Structure

- `daily-total-female-births.csv`: Input dataset
- `main.py`: Main script containing all the code
- `models/`: Directory containing saved models
  - `prophet_model.pkl`: Saved Prophet model
  - `arima_model.pkl`: Saved ARIMA model
  - `lstm_model.h5`: Saved LSTM model
  - `scaler.pkl`: Saved MinMaxScaler for LSTM

## Usage

1. Ensure you have all the required libraries installed.
2. Place the `daily-total-female-births.csv` file in the same directory as the script.
3. Run the main script:

```
python main.py
```

## Model Comparison

The script will train and evaluate three different models:

1. **Prophet**: A procedure for forecasting time series data based on an additive model.
2. **ARIMA**: A statistical model for analyzing and forecasting time series data.
3. **LSTM**: A type of recurrent neural network capable of learning long-term dependencies.

The script will output performance metrics (MAE, MSE, RMSE) for each model and generate visualizations comparing the forecasts.

## Visualizations

The script generates several plots:

1. Original data plot
2. Prophet forecast and components
3. ARIMA forecast
4. LSTM predictions
5. Comparison plot of all three models

## Saved Models

After training, the models are saved in the `models/` directory for future use.

## Results

The script will print a summary of the performance metrics for each model, allowing for easy comparison.

## Future Improvements

- Implement hyperparameter tuning for each model
- Explore ensemble methods combining multiple models
- Add more advanced deep learning models (e.g., Transformer-based models)
- Incorporate external factors that might influence birth rates
