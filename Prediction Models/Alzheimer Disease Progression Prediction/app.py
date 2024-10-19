import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = pickle.load(open("disease_progression_model.pkl", 'rb'))  # Update based on best model

# Define input fields
def user_input_features():
    age = st.number_input("Age", min_value=50, max_value=90, value=65)
    gender = st.selectbox("Gender", ("Male", "Female"))
    years_since_diagnosis = st.number_input("Years Since Diagnosis", min_value=0, max_value=15, value=5)
    lesion_volume = st.slider("Lesion Volume (cm³)", 0.0, 50.0, 10.0)
    hippocampal_volume = st.slider("Hippocampal Volume (cm³)", 2.0, 4.5, 3.0)
    white_matter_changes = st.slider("White Matter Changes (cm³)", 0.0, 10.0, 2.0)
    cognitive_score = st.slider("Cognitive Score", 0, 30, 20)
    memory_score = st.slider("Memory Score", 0, 10, 5)
    disease_stage = st.selectbox("Disease Stage", ("Early", "Moderate", "Advanced"))

    # Encode categorical values
    gender = 0 if gender == "Male" else 1
    disease_stage = {"Early": 0, "Moderate": 1, "Advanced": 2}[disease_stage]

    # Create a feature array
    features = np.array([age, gender, years_since_diagnosis, lesion_volume, hippocampal_volume, white_matter_changes, 
                         cognitive_score, memory_score, disease_stage]).reshape(1, -1)
    return features

# App header
st.title("Neurodegenerative Disease Progression Predictor")

# Collect user input
input_features = user_input_features()

# Scale the input using StandardScaler
scaler = StandardScaler()
input_features = scaler.fit_transform(input_features)

# Make a prediction
if st.button("Predict Progression"):
    prediction = model.predict(input_features)
    progression_status = "Progressed" if prediction == 1 else "Early Progression"
    st.write(f"The predicted disease progression status is: **{progression_status}**")
