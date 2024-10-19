# MS Treatment Response Predictor

## Project Overview
The **MS Treatment Response Predictor** is a machine learning-based web application designed to predict the efficacy of different treatments for Multiple Sclerosis (MS). Using a set of patient input data like age, gender, disease duration, EDSS score, and more, the model predicts how effective a particular treatment will be for the patient. This project serves as a valuable tool for both healthcare professionals and patients in choosing the most effective MS treatment.

## Problem Summary
Multiple Sclerosis (MS) is a chronic disease affecting the central nervous system. Treatment effectiveness varies from patient to patient, depending on various factors like age, gender, disease duration, MRI lesions, and more. This project addresses the challenge of predicting which treatment will be most effective for a specific patient, helping to personalize the treatment plan.

### Solution
We have built a machine learning model trained on a dataset containing various MS patient details and treatment responses. The model takes in inputs like age, gender, EDSS score, disease duration, etc., and predicts the efficacy of the chosen treatment.

### Technologies Used:
- **Python**: for data processing and model development
- **Pandas**: for data manipulation
- **Pickle**: for loading the trained model
- **Streamlit**: for building the web app interface

## Installation Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/UppuluriKalyani/ML-Nexus.git
    cd ML-Nexus
    cd Prediction Models
    cd Treatment Response Predictor
    ```

2. **Install required packages**:
    Ensure that you have Python installed on your system. Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    Start the Streamlit web app by running:
    ```bash
    streamlit run app.py
    ```

## Usage Instructions

1. **Input Patient Data**: The web app allows you to input the following patient details:
   - Age (between 18 to 100)
   - Gender (Male or Female)
   - Disease duration (1 to 50 years)
   - EDSS score (0.0 to 7.5)
   - Number of previous relapses
   - MRI lesions (0 to 50)
   - Treatment type
   - Treatment duration (1 to 60 months)
   - Side effects severity (0 to 5)

2. **Predict Efficacy**: After filling in the data, click the "Predict Efficacy" button to get a prediction of how effective the treatment will be for the patient.

## Examples
After running the app, here is an example of the input and output:

**Input**:
- Age: 45
- Gender: Female
- Disease duration: 10 years
- EDSS score: 4.0
- Previous relapses: 2
- MRI lesions: 15
- Treatment: Ocrelizumab
- Treatment duration: 12 months
- Side effects: 3

**Predicted Efficacy**: 0.85 (85% effectiveness)

## Screenshot
![app](https://github.com/user-attachments/assets/b24d1548-1a82-49f0-aa00-8487e8bcab11)