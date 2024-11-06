# Advertisement Budget Prediction

## Project Overview
This project aims to predict sales revenue based on advertisement budgets allocated across different channels, specifically TV, radio, and newspapers. Using multiple machine learning models, we analyze the impact of each media on overall sales, helping businesses make data-driven decisions on advertisement spending.

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/), which includes the following features:
- **TV**: Budget spent on TV advertising.
- **Radio**: Budget spent on radio advertising.
- **Newspaper**: Budget spent on newspaper advertising.
- **Sales**: Sales revenue generated.

## Project Structure
- **Data Analysis**: Preliminary analysis to understand relationships between features and target variable.
- **Data Cleaning**: Handling missing values and any data preprocessing required.
- **Model Building**: Creating regression models to predict sales, including:
  - **Multiple Linear Regression**
  - **Polynomial Regression**
  - **Ridge Regression**
  - **Lasso Regression**
- **Model Evaluation**: Models are evaluated using:
  - **R-squared**
  - **Root Mean Squared Error (RMSE)**
  - **Residual Sum of Squares (RSS)**
  - **Mean Squared Error (MSE)**

## Evaluation Results
Models are compared based on the evaluation metrics above, identifying the best fit based on performance on both training and testing data. Polynomial regression models were found to have the best performance based on the RMSE and R-squared metrics.

