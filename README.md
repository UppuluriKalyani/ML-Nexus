A Machine Learning end to end model for predicting flight fare based on a highly imbalanced dataset ,which upon performing EDA operations and creating and the model gives a high accuracy of price predicted across different flight routes

## Dataset Link: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh

Flight Price Prediction:
-------------------------
This project focuses on predicting flight prices using machine learning techniques, specifically using RandomForestRegressor from scikit-learn. The goal is to predict the fare of flights based on various features such as airline, source, destination, duration, etc.

Table of Contents:
--------------------
Dataset
Installation
Usage
File Descriptions
Methodology
Results
Future Improvements
Contact
Dataset
The dataset used in this project contains information about flight details such as airline, source, destination, duration, total stops, and price. It was obtained from Kaggle and consists of both categorical and numerical features; dataset to be uploaded in the repository directory or can be directly imported from Kaggle

Installation:
--------------
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/your-username/flight-price-prediction.git
cd flight-price-prediction
Install the required Python packages using pip:

pip install -r requirements.txt
Ensure you have Python (>=3.6) installed on your system.

Usage:
------
Data Preparation: Ensure your dataset (dataset.csv) is placed in the root directory.(Here, we have used flight.csv dataset)

Model Training: Execute train.py to train the RandomForestRegressor model.


python train.py
This script will load the dataset, preprocess the data, train the model, and save it as flight_rf.pkl.

Prediction: Use the trained model for predictions with predict.py.


python predict.py
This script loads the trained model and predicts flight prices based on user inputs.

Web Application (Optional): Run the web application to interactively predict flight prices using Streamlit.


streamlit run app.py
Open a browser and go to http://localhost:8501 (or port name as defined; most commonly used is port 5000) to use the application.

File Descriptions:
-------------------
data.csv: Dataset containing flight details.
train.py: Python script to preprocess data, train the RandomForestRegressor model, and save it.
predict.py: Python script to load the trained model and predict flight prices based on user inputs.
app.py: Streamlit web application for interactive flight price prediction.

Methodology:
------------
Data Preprocessing: Cleaning data, handling missing values, encoding categorical variables, and scaling numerical features.

Feature Selection: Using ExtraTreesRegressor to determine feature importances and selecting relevant features for training.

Model Selection: RandomForestRegressor is chosen for its ensemble learning technique and ability to handle complex datasets.

Model Training: Hyperparameter tuning using RandomizedSearchCV to optimize the model's performance.

Model Evaluation: Metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score are used to evaluate the model.

Results:
---------
After training and evaluating the RandomForestRegressor model:

MAE: 1165.61
MSE: 4062650.69
RMSE: 2015.60
R-squared score: 0.8117
The model shows promising results in predicting flight prices based on the selected features.

Future Improvements:
----------------------
Feature Engineering: Explore additional features that could enhance model performance.
Algorithm Tuning: Further fine-tuning of hyperparameters to improve predictive accuracy.
Deployment: Deploy the model as a web service using Flask or Streamlit or Docker or integrate into a production environment ( here we used StreamLit). More improvements and deployment as a indpendent web application is in progress.
