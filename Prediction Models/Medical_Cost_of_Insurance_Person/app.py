import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    return data

data = load_data()

# Preprocess the data
lab = LabelEncoder()
data['sex'] = lab.fit_transform(data['sex'])
data['smoker'] = lab.fit_transform(data['smoker'])
data['region'] = lab.fit_transform(data['region'])

# Split the data
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': tree.DecisionTreeRegressor(),
    'Support Vector Regressor': SVR(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0)
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Streamlit app interface
st.title("Insurance Cost Prediction")

# User input for prediction
age = st.number_input('Age', min_value=1, max_value=100, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Encoding user input
sex = 1 if sex == 'male' else 0
smoker = 1 if smoker == 'yes' else 0
region = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}[region]

# Input vector for prediction
user_input = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -6)

# Model selection
model_choice = st.selectbox('Choose Regression Model', list(models.keys()))

if st.button('Predict'):
    model = models[model_choice]
    prediction = model.predict(user_input)
    st.write(f'Predicted Insurance Charges: ${int(prediction[0]):,}')
    
    # Display r2 score
    y_pred = model.predict(X_test)
    st.write(f"{model_choice} r2_score: {r2_score(y_test, y_pred):.2f}")
