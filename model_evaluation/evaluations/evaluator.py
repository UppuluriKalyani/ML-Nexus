from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
import os

class ModelEvaluator:
    def evaluate(self, model_name, dataset_name):
        model_path = f'models/{model_name}.pkl'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found.")
            return None
        
        model = joblib.load(model_path)  # Load model
        X, y = self.load_data(dataset_name)  # Load data

        if X is None or y is None:
            return None

        y_pred = model.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted')
        }
        return metrics

    def load_data(self, dataset_name):
        # Implement data loading based on the dataset type
        dataset_path = os.path.join('datasets', dataset_name)
        if dataset_name.endswith('.csv'):
            data = pd.read_csv(dataset_path)
            X = data.iloc[:, :-1].values  # Features (all but last column)
            y = data.iloc[:, -1].values   # Labels (last column)
            return X, y
        # Add more conditions to load other dataset types if necessary
        return None, None
