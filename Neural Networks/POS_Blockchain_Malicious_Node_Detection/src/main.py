import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from utils import plot_training_history, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from model import create_lstm_model
import os
from tensorflow.keras.utils import to_categorical

def main():
    
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    
    data = pd.read_csv("../data/Blockchain101.csv")
    data.fillna(method='ffill', inplace=True)

    
    label_encoder = LabelEncoder()
    data['TxnFee (Binary)'] = label_encoder.fit_transform(data['TxnFee (Binary)'])

    target = 'Node Label'
    features = data.columns.drop(target)

    
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    X = data[features].values
    y = label_encoder.fit_transform(data[target].values)
    n_classes = len(np.unique(y))
    y_categorical = to_categorical(y)

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    histories = []
    all_y_test = []
    all_y_pred = []
    all_y_pred_proba = []

    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"Training Fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y_categorical[train_index], y_categorical[val_index]

    
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        input_shape = (1, X.shape[1])

    
        model = create_lstm_model(input_shape, n_classes)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                          validation_data=(X_val, y_val), verbose=2)
        histories.append(history)

        # Make predictions
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)

        all_y_test.extend(np.argmax(y_val, axis=1))
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_proba)

    
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    all_y_pred_proba = np.array(all_y_pred_proba)

    
    plot_training_history(histories, 'LSTM')
    plot_roc_curve(to_categorical(all_y_test), all_y_pred_proba, 'LSTM')
    plot_pr_curve(to_categorical(all_y_test), all_y_pred_proba, 'LSTM')
    
    
    classification_rep = classification_report(all_y_test, all_y_pred)
    cm = confusion_matrix(all_y_test, all_y_pred)
    plot_confusion_matrix(cm, 'LSTM')

    
    with open('results/metrics/lstm_metrics.txt', 'w') as f:
        f.write("LSTM Model Classification Report:\n\n")
        f.write(classification_rep)

    print("Training completed. Results saved in results directory.")

if __name__ == "__main__":
    main()