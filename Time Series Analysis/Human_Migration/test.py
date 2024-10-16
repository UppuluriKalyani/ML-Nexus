# test.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import warnings


warnings.filterwarnings("ignore")
# Load models and scaler from the saved directory
scaler = joblib.load('models/scaler.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
gb_model = joblib.load('models/gradient_boosting_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')

print("Models and scaler loaded successfully!")

# Load and preprocess the dataset
data = pd.read_csv('migration_nz.csv')

# Replace string labels with numeric codes
data['Measure'].replace({"Arrivals": 0, "Departures": 1, "Net": 2}, inplace=True)
data['CountryID'] = pd.factorize(data['Country'])[0]
data['CitID'] = pd.factorize(data['Citizenship'])[0]
data['Value'].fillna(data['Value'].median(), inplace=True)

# Drop unnecessary columns
data.drop(['Country', 'Citizenship'], axis=1, inplace=True)

# Prepare features with Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(data[['CountryID', 'Measure', 'Year', 'CitID']])

# Define target
Y = data['Value']

# Train-Test Split (Ensure same random state for consistency)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.3, random_state=42)

# Use the saved scaler to transform the test data
X_test = scaler.transform(X_test)

# Predict and evaluate with the loaded models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model.__class__.__name__} - R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

# Evaluate RandomForest, GradientBoosting, and XGBoost models
evaluate_model(rf_model, X_test, y_test)
evaluate_model(gb_model, X_test, y_test)
evaluate_model(xgb_model, X_test, y_test)
