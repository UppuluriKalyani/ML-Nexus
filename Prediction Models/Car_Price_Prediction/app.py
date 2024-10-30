# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Title and description
st.title('Car Price Prediction App')
st.write("""
This app predicts the **Selling Price of Used Cars** based on car features using Linear Regression and Lasso Regression models.
""")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("car data.csv")
    return data

data = load_data()

# Sidebar for user input features
st.sidebar.header('User Input Features')

def user_input_features():
    Year = st.sidebar.slider('Year of Manufacture', 2000, 2020, 2015)
    Present_Price = st.sidebar.slider('Present Price (in lakhs)', 1.0, 50.0, 10.0)
    Kms_Driven = st.sidebar.slider('Kms Driven', 500, 200000, 50000)
    Fuel_Type = st.sidebar.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
    Seller_Type = st.sidebar.selectbox('Seller Type', ('Dealer', 'Individual'))
    Transmission = st.sidebar.selectbox('Transmission', ('Manual', 'Automatic'))
    Owner = st.sidebar.slider('Number of Owners', 0, 3, 0)
    
    fuel_type_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_type_dict = {'Dealer': 0, 'Individual': 1}
    transmission_dict = {'Manual': 0, 'Automatic': 1}
    
    data = {'Year': Year,
            'Present_Price': Present_Price,
            'Kms_Driven': Kms_Driven,
            'Fuel_Type': fuel_type_dict[Fuel_Type],
            'Seller_Type': seller_type_dict[Seller_Type],
            'Transmission': transmission_dict[Transmission],
            'Owner': Owner}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader('User Input Features')
st.write(input_df)

# Preprocess the data
def preprocess_data(data):
    data.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
    data.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
    data.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
    X = data.drop(columns=['Car_Name', 'Selling_Price'])
    y = data['Selling_Price']
    return X, y

X, y = preprocess_data(data)

# Train models
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Lasso Regression model
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

# Make predictions based on user inputs
lr_pred = lr_model.predict(input_df)
lasso_pred = lasso_model.predict(input_df)

# Display predictions
st.subheader('Predicted Selling Price (Linear Regression)')
st.write(f'₹ {lr_pred[0]:.2f} lakhs')

st.subheader('Predicted Selling Price (Lasso Regression)')
st.write(f'₹ {lasso_pred[0]:.2f} lakhs')

# Plot Actual vs Predicted (Linear Regression)
st.subheader('Linear Regression: Actual vs Predicted Values on Training Set')
y_train_pred = lr_model.predict(x_train)
plt.figure(figsize=(6,4))
plt.scatter(y_train, y_train_pred, color='blue')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Linear Regression: Actual vs Predicted')
st.pyplot(plt)

# Plot Actual vs Predicted (Lasso Regression)
st.subheader('Lasso Regression: Actual vs Predicted Values on Training Set')
y_train_pred_lasso = lasso_model.predict(x_train)
plt.figure(figsize=(6,4))
plt.scatter(y_train, y_train_pred_lasso, color='green')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Lasso Regression: Actual vs Predicted')
st.pyplot(plt)
