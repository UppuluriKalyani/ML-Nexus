# Bank Deposit Prediction Using Machine Learning

## Project Overview

This project utilizes machine learning models to predict the likelihood of a bank client subscribing to a term deposit after a marketing campaign. By analyzing various client features such as age, job, balance, and previous campaign interactions, the project aims to improve targeting and conversion rates for future campaigns.

## Problem Statement

Predicting customer deposit subscription enables banks to optimize marketing strategies and direct efforts towards high-potential clients, thus enhancing both efficiency and campaign effectiveness. This project addresses the challenge of identifying clients who are more likely to subscribe to a term deposit.

## Data

The dataset contains client information including:

- **Demographics:** Age, job, marital status, and education level.
- **Financial Information:** Account balance and loan details.
- **Campaign Interaction:** Contact type, number of contacts, days since previous contact, etc.

## Approach

1. **Exploratory Data Analysis (EDA):** Analyze and visualize the dataset to understand distributions, relationships, and feature relevance.
2. **Data Preprocessing:** Handle missing values, encode categorical variables, and scale features as needed.
3. **Model Building:** Implement multiple machine learning models, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting
4. **Evaluation:** Assess models based on accuracy, precision, recall, and F1-score to determine the most effective model for predicting term deposit subscription.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
