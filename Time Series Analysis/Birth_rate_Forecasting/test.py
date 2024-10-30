import unittest
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

class TestNotebookFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data
        cls.df = pd.read_csv("daily-total-births.csv", parse_dates=['date'], date_parser=pd.to_datetime)
        cls.df.columns = ['ds', 'y']

        # Load models
        with open("models/prophet_model.pkl", 'rb') as f:
            cls.prophet_model = pickle.load(f)
        
        with open("models/arima_model.pkl", 'rb') as f:
            cls.arima_model = pickle.load(f)
        
        cls.lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
        
        with open("models/scaler.pkl", 'rb') as f:
            cls.scaler = pickle.load(f)

    def test_data_loading(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(self.df.columns.tolist(), ['ds', 'y'])
        self.assertGreater(len(self.df), 0)

    def test_prophet_model(self):
        future = self.prophet_model.make_future_dataframe(periods=50, freq='D')
        forecast = self.prophet_model.predict(future)
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertGreater(len(forecast), len(self.df))

    def test_arima_model(self):
        arima_forecast = self.arima_model.forecast(steps=50)
        self.assertEqual(len(arima_forecast), 50)

    def test_lstm_model(self):
        # Prepare test data
        look_back = 3
        test_data = self.scaler.transform(self.df[['y']].values[-look_back:])
        test_data = test_data.reshape((1, look_back, 1))
        
        # Make prediction
        lstm_pred = self.lstm_model.predict(test_data)
        
        self.assertEqual(lstm_pred.shape, (1, 1))

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists("models/prophet_model.pkl"))
        self.assertTrue(os.path.exists("models/arima_model.pkl"))
        self.assertTrue(os.path.exists("models/lstm_model.h5"))
        self.assertTrue(os.path.exists("models/scaler.pkl"))

if __name__ == '__main__':
    unittest.main()