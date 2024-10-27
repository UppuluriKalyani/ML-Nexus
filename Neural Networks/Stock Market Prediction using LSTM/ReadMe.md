## Stock Market Prediction using LSTM

This repository contains code for predicting stock market prices using Long Short-Term Memory (LSTM) neural networks. LSTM is a type of recurrent neural network (RNN) known for its effectiveness in handling sequential data, making it suitable for time series forecasting tasks such as stock price prediction.

### Requirements
- Python 3.x
- Libraries: numpy, matplotlib, pandas, pandas_datareader, scikit-learn, tensorflow, yfinance

### Instructions
1. **Fetching Data**: Historical stock market data is fetched using the Yahoo Finance API through the `yfinance` library.

2. **Data Preprocessing**: The fetched data is preprocessed by scaling it using Min-Max scaling to ensure all values are between 0 and 1. The data is split into training and testing sets, and further processed to be suitable for LSTM input.

3. **Model Building**: An LSTM neural network model is built using TensorFlow's Keras API. The model architecture consists of multiple LSTM layers followed by dropout layers to prevent overfitting.

4. **Model Training**: The LSTM model is trained using the training data with a specified number of epochs and batch size.

5. **Prediction**: After training, the model is used to predict future stock market prices. Predictions are made based on the last few days of historical data.

6. **Visualization**: Predicted prices are plotted alongside actual prices using matplotlib to visualize the model's performance.

### Note
- Ensure you have sufficient historical data for accurate prediction.
- The model performance may vary based on factors such as market dynamics and the chosen hyperparameters.



