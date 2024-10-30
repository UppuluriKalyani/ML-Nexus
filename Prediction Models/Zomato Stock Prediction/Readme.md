
# Zomato Stock Price Prediction 

This project is focused on predicting the stock prices of Zomato using a Long Short-Term Memory (LSTM) neural network implemented in TensorFlow. The model is trained on historical stock price data and aims to predict future stock prices based on past trends.

## Project Structure

The project is organized as follows:

- **Data Preprocessing**: 
  The data is loaded and preprocessed by normalizing the stock prices and splitting it into training and testing datasets.
  
- **Model Construction**: 
  A Sequential model is built using LSTM layers to capture the sequential nature of stock price data.

- **Model Training**: 
  The LSTM model is trained on the processed data using a defined sequence length, and the EarlyStopping callback is used to prevent overfitting.

## Requirements

The following Python libraries are required to run the project:

- `tensorflow`
- `sklearn`
- `matplotlib`
- `pandas`
- `numpy`

## Data

The dataset used in this project includes historical Zomato stock prices. The important features used in the prediction process include:

- **Date**: The date of the stock price data.
- **Close**: The closing stock price of Zomato on that day.

The data is preprocessed as follows:
- **Normalization**: Stock prices are normalized using MinMaxScaler to scale values between 0 and 1.
- **Sequence Creation**: Time series sequences of stock prices are created, with each sequence consisting of 60 days of historical data used to predict the stock price of the 61st day.

## Model Architecture

The LSTM model has the following architecture:

1. **LSTM Layer**: Extracts time dependencies from the input data.
2. **Dense Layer**: Maps the LSTM outputs to the target variable (future stock price).

The model uses the following configuration:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam Optimizer

## How to Run

1. Clone the repository or download the notebook.
2. Install the required libraries.
   ```bash
   pip install tensorflow sklearn matplotlib pandas numpy
   ```
3. Download the dataset and ensure it is available in the correct path.
4. Run the notebook to train the model and visualize predictions.

## Key Functions

- `create_sequences(data, seq_length)`: This function creates sequences from the data for LSTM input.
- **Model Training**: The model is trained on 80% of the data, while 20% is used for testing.
