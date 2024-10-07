import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Predictions
    This app predicts the **Iris Flower** Type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 0.1, 7.9, 5.3)
    sepal_width = st.sidebar.slider('Sepal Width', 0.1, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 0.1, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    features = pd.DataFrame(data, index=[0])

    return features


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

rfc = RandomForestClassifier()
rfc.fit(X=X, y=y)

prediction = rfc.predict(df)
prediction_prob = rfc.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)