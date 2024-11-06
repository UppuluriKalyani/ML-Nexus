import joblib

def load_model(model_name):
    return joblib.load(f'models/{model_name}.pkl')

def save_model(model, model_name):
    joblib.dump(model, f'models/{model_name}.pkl')
