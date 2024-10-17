import streamlit as st
import pickle
import numpy as np
import datetime as dt

# Load the saved model
with open('bigmart_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get the current year for input validation
current_year = dt.datetime.today().year

# Define the prediction function
def predict_sales(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_establishment_year):
    # Dictionaries for encoding categorical inputs
    outlet_dict = {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4,
                   'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}
    size_dict = {'High': 0, 'Medium': 1, 'Small': 2}
    type_dict = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}

    # Convert categorical inputs to numerical values
    p2 = outlet_dict[outlet_identifier]
    p3 = size_dict[outlet_size]
    p4 = type_dict[outlet_type]
    p5 = current_year - outlet_establishment_year

    # Make prediction using the model
    result = model.predict(np.array([[item_mrp, p2, p3, p4, p5]]))
    return result[0]

# Streamlit UI setup
st.title("Big Mart Sales Prediction Web App")
st.write("Enter the details below to predict the sales for a product.")

# Input fields for the user
item_mrp = st.number_input("Enter Item MRP", min_value=0.0, step=0.1)
outlet_identifier = st.selectbox("Select Outlet Identifier",
                                 ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
                                  'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
outlet_size = st.selectbox("Select Outlet Size", ['High', 'Medium', 'Small'])
outlet_type = st.selectbox("Select Outlet Type",
                           ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
outlet_establishment_year = st.number_input("Enter Outlet Establishment Year",
                                            min_value=1900, max_value=current_year, step=1)

# Predict button
if st.button('Predict'):
    # Perform the prediction
    predicted_sales = predict_sales(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_establishment_year)

    # Display the result with a margin for prediction range
    st.success(f"Predicted Sales Amount: {predicted_sales - 714.42:.2f} to {predicted_sales + 714.42:.2f}")
