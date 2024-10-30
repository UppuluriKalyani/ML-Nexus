import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    data.to_csv(f'data/{ticker}_data.csv')
    return data


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


if __name__ == "__main__":
    
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    seq_length = 60
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    
    data = load_and_preprocess_data(ticker, start_date, end_date)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    
    X, y = create_sequences(scaled_data, seq_length)

    
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size-seq_length], X[train_size-seq_length:]
    y_train, y_test = y[:train_size-seq_length], y[train_size-seq_length:]

    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    
    model = StockLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train).numpy()
        test_predictions = model(X_test).numpy()

    
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

    
    np.savetxt('data/train_predictions.csv', train_predictions, delimiter=',')
    np.savetxt('data/test_predictions.csv', test_predictions, delimiter=',')

    print("Model training complete. Predictions saved to data/train_predictions.csv and data/test_predictions.csv")

    
    plt.figure(figsize=(12,6))
    plt.plot(data.index[train_size:], data.values[train_size:], label='Actual Prices')
    plt.plot(data.index[train_size:], test_predictions, label='Predicted Prices')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('results/price_prediction.png')
    plt.close()
