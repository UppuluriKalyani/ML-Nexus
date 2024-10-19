# Neurodegenerative Disease Progression Predictor

## Project Overview
The Neurodegenerative Disease Progression Predictor is a machine learning application designed to predict the progression of neurodegenerative diseases based on clinical and demographic features. This tool aids healthcare professionals in early detection and intervention.

## Code Details
The application is built using Python and the Streamlit library for the user interface. The key components of the code include:

- **Model Loading**: The trained model is loaded using `pickle`, allowing for predictions based on user input.
- **User Input Features**: A function `user_input_features()` collects and processes input data such as age, gender, years since diagnosis, and various health metrics.
- **Data Scaling**: The input features are scaled using `StandardScaler` to ensure consistent input for the model.
- **Prediction Logic**: Upon clicking the "Predict Progression" button, the app makes a prediction based on the input features and displays the progression status.

## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/UppuluriKalyani/ML-Nexus.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd ML-Nexus/Prediction Models/Alzheimer Disease Progression Prediction
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

1. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

2. **Input Features:**
   Fill in the required fields and click "Predict Progression" to obtain the predicted disease progression status.

## Screenshot
![Screenshot 2024-10-19 005941](https://github.com/user-attachments/assets/3fe24058-91e4-435d-b4f5-067dee3c844b)