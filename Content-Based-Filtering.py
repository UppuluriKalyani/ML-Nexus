import pandas as pd

def recommend_based_on_data_type(data, task_type):
    # Check the type of dataset
    if isinstance(data, pd.DataFrame):
        # Tabular data
        if task_type == 'classification':
            return ['Random Forest', 'Logistic Regression', 'XGBoost']
        elif task_type == 'regression':
            return ['Linear Regression', 'Random Forest Regressor', 'XGBoost']
    elif 'image' in data.lower():
        # Image data
        return ['CNN', 'ResNet', 'EfficientNet']
    elif 'text' in data.lower():
        # Text data
        if task_type == 'classification':
            return ['RNN', 'LSTM', 'BERT']
        elif task_type == 'generation':
            return ['GPT-3', 'Transformer']
    else:
        return ['Please provide a supported data type (tabular, image, or text).']

# Example usage:
dataset_type = 'image'
task_type = 'classification'
models = recommend_based_on_data_type(dataset_type, task_type)
print(f"Recommended models for {dataset_type} and {task_type}: {models}")
