import streamlit as st
import pickle
import pandas as pd  # Import pandas to handle DataFrames
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the model and the scaler
model_path = 'Prediction Models/sleep_disorder_predictor/saved_models/Model_Prediction.sav'
preprocessor_path = 'Prediction Models/sleep_disorder_predictor/saved_models/preprocessor.sav'

# Load the pre-trained model and scaler using pickle
loaded_model = pickle.load(open(model_path, 'rb'))
preprocessor = pickle.load(open(preprocessor_path, 'rb'))

# Define the prediction function
def disease_get_prediction(Age, Sleep_Duration,  
                            Heart_Rate, Daily_Steps, 
                            Systolic, Diastolic, Occupation, Quality_of_Sleep, Gender, 
                            Physical_Activity_Level, Stress_Level, BMI_Category):
    # Create a DataFrame with the features using correct column names
    features = pd.DataFrame({
        'Age': [int(Age)],
        'Sleep Duration': [float(Sleep_Duration)],  # Changed to match expected name
        'Heart Rate': [int(Heart_Rate)],            # Changed to match expected name
        'Daily Steps': [int(Daily_Steps)],          # Changed to match expected name
        'Systolic': [float(Systolic)],
        'Diastolic': [float(Diastolic)],
        'Occupation': [Occupation],
        'Quality of Sleep': [int(Quality_of_Sleep)], # Changed to match expected name
        'Gender': [Gender],
        'Physical Activity Level': [int(Physical_Activity_Level)], # Changed to match expected name
        'Stress Level': [int(Stress_Level)],       # Changed to match expected name
        'BMI Category': [BMI_Category]               # Changed to match expected name
    })

    # Apply the preprocessor (make sure it expects a DataFrame)
    preprocessed_data = preprocessor.transform(features)

    # Make prediction
    prediction = loaded_model.predict(preprocessed_data)

    return prediction
