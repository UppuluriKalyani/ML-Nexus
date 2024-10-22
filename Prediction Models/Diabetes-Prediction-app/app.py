# Import necessary libraries
# Importing the basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained Gradient Boosting model
model = joblib.load('gradient_boosting_model.pkl')  # Replace 'gradient_boosting_model.pkl' with your actual model file

# Function to preprocess input data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Streamlit App
def main():
    # Title and description
    st.title("Diabetes Prediction App")
    st.write("This app predicts whether a person has diabetes based on selected features.")

    # User input for features
    pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.slider("Skinfold Thickness", 0, 99, 23)
    insulin = st.slider("2-Hour Serum Insulin", 0, 846, 79)
    bmi = st.slider("Body Mass Index (BMI)", 0.0, 67.1, 32.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.372)
    age = st.slider("Age", 21, 81, 29)

    # Create a dataframe with user input
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    # Preprocess the input data
    input_data_scaled = preprocess_data(input_data)

    # Display the input data & Predict the outcome
    st.subheader("Input Data:")
    st.write(input_data)
    prediction = model.predict(input_data_scaled)
    # Display the prediction
    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.write("The model predicts that the person does not have diabetes.")
    else:
        st.write("The model predicts that the person has diabetes.")

if __name__ == "__main__":
    main()
