# Play Store Ratings Prediction

## Overview
This project uses machine learning regression models to predict app ratings on the Google Play Store. By analyzing the relationship between app attributes (e.g., category, reviews, size, type, etc.) and ratings, this model aids in understanding what factors contribute to higher app ratings.

## Problem Statement
The goal of this project is to predict an app's rating using available metadata and features, such as category, size, and price. This prediction model provides insights into key factors that influence app ratings, offering app developers valuable information for enhancing app quality and user satisfaction.

## Dataset
The dataset includes information on:
- App name
- Category
- Reviews count
- Size
- Installs count
- Type (Free/Paid)
- Price
- Content Rating
- Genres
- Last updated date
- Current version
- Android version compatibility

## Objectives
1. **Data Preprocessing**: Handle missing values, encode categorical variables, and normalize the data where necessary.
2. **Exploratory Data Analysis**: Perform statistical analysis to uncover relationships between the features and the target variable (app rating).
3. **Model Training**: Develop multiple regression models and evaluate them to determine the best-fit model.
4. **Evaluation**: Assess model performance using metrics such as R², Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Project Structure
- `data/`: Contains the Play Store dataset.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, and model training.
- `models/`: Saved trained models for predictions.
- `src/`: Scripts for data processing and model building.
