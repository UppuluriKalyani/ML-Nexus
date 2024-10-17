import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Set up the title and sidebar
st.title("House Price Prediction App")
st.sidebar.title("Model Configuration")

# File upload for training dataset
st.sidebar.subheader("Upload Dataset")
train_file = st.sidebar.file_uploader("Upload your train CSV file", type=["csv"])

if train_file is not None:
    # Load the training data
    train_data = pd.read_csv(train_file)

    # Data Preprocessing
    train_data = train_data.select_dtypes(include=['int64', 'float64']).dropna()
    x_train = train_data.drop(columns=['SalePrice'], axis=1)
    y_train = pd.DataFrame(train_data['SalePrice']).dropna()

    # Display data information
    if st.sidebar.checkbox("Show Train Data Info"):
        st.write("Train Data Info:")
        st.write(x_train.info())
        st.write("Train Data Head:")
        st.write(train_data.head())

    # Model Selection
    model_type = st.sidebar.selectbox("Select Model", ("Linear Regression", "Logistic Regression", "Polynomial Regression"))

    # Train the model based on user selection
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(x_train, y_train)
        st.subheader("Linear Regression Model")

    elif model_type == "Logistic Regression":
        model = LogisticRegression()
        model.fit(x_train, y_train)
        st.subheader("Logistic Regression Model")

    elif model_type == "Polynomial Regression":
        degree = st.sidebar.slider("Select degree of polynomial", 2, 5, 2)
        poly = PolynomialFeatures(degree=degree)
        x_poly_train = poly.fit_transform(x_train)
        model = LinearRegression()
        model.fit(x_poly_train, y_train)
        st.subheader(f"Polynomial Regression Model (Degree: {degree})")

    # Collect input from the user for prediction
    st.sidebar.subheader("Input Features for Prediction")
    input_values = {}
    for col in x_train.columns:
        input_values[col] = st.sidebar.number_input(f"Input {col}", min_value=float(x_train[col].min()), max_value=float(x_train[col].max()), value=float(x_train[col].mean()))

    # Convert input values into a DataFrame for prediction
    input_data = pd.DataFrame([input_values])

    # Predict based on user input
    if model_type == "Polynomial Regression":
        input_data_poly = poly.transform(input_data)
        y_pred = model.predict(input_data_poly)
    else:
        y_pred = model.predict(input_data)

    st.write("Prediction based on input features:")
    st.write(y_pred[0])

    # Visualize the prediction on the train data
    if model_type != "Polynomial Regression":
        y_pred_train = model.predict(x_train)
        r2 = r2_score(y_train, y_pred_train)
        st.write(f"R² score on training data: {r2}")

        plt.scatter(y_train, y_pred_train)
        plt.plot([min(y_pred_train), max(y_pred_train)], [min(y_pred_train), max(y_pred_train)], color='red', linestyle='--')  # Line of best fit
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values (Train)')
        st.pyplot(plt)

# Footer
st.sidebar.text("Developed with Streamlit")
