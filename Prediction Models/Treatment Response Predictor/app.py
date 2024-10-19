import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('ms_treatment_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('MS Treatment Response Predictor')

# Create input fields
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['M', 'F'])
disease_duration = st.number_input('Disease Duration (years)', min_value=1, max_value=50)
edss_score = st.slider('EDSS Score', 0.0, 7.5, 2.5, 0.5)
previous_relapses = st.number_input('Previous Relapses', min_value=0, max_value=10)
mri_lesions = st.number_input('MRI Lesions', min_value=0, max_value=50)
treatment = st.selectbox('Treatment', [
    "Interferon beta-1a", "Interferon beta-1b", "Glatiramer acetate", 
    "Fingolimod", "Natalizumab", "Ocrelizumab", "Dimethyl fumarate",
    "Teriflunomide", "Alemtuzumab", "Cladribine", "Siponimod"
])
treatment_duration = st.number_input('Treatment Duration (months)', min_value=1, max_value=60)
side_effects = st.slider('Side Effects Severity', 0, 5, 1)

# Create a dataframe from inputs
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'disease_duration': [disease_duration],
    'edss_score': [edss_score],
    'previous_relapses': [previous_relapses],
    'mri_lesions': [mri_lesions],
    'treatment': [treatment],
    'treatment_duration': [treatment_duration],
    'side_effects': [side_effects]
})

# Make prediction
if st.button('Predict Efficacy'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Efficacy: {prediction[0]}')