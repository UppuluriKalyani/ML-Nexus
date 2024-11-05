# Mobile Price Prediction using Classification Models

This repository contains machine learning models designed to classify mobile phones into different price ranges based on their specifications. The project leverages multiple classification algorithms to predict the price category, helping users understand the factors that influence mobile pricing.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Classification Models](#classification-models)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)


## Problem Statement

With numerous features influencing mobile phone prices, it can be challenging to predict a mobile phone's price range accurately. This project aims to build a classification model that predicts the price category of a mobile phone based on its specifications, providing a quick and reliable tool for price estimation.

## Project Overview

This project involves:
1. **Data Analysis and Visualization**: Understanding and visualizing the key features that impact mobile pricing.
2. **Model Training**: Building and comparing several classification models, including logistic regression, decision trees, and support vector machines.
3. **Evaluation**: Assessing the models' performance to identify the most accurate classification approach.

The model assists users in predicting the price range based on technical specifications, improving the transparency of mobile phone pricing.

## Classification Models

The project evaluates multiple classification algorithms:
- **Logistic Regression**: For binary classification and baseline comparison.
- **Decision Trees**: Capturing non-linear relationships in the data.
- **Support Vector Machines (SVM)**: Finding optimal hyperplanes for classification.
- **Random Forest**: An ensemble method improving accuracy with multiple decision trees.

## Dataset

The dataset includes various mobile phone features such as:
- **RAM**: Memory capacity.
- **Battery Power**: Battery life.
- **Pixel Resolution**: Display quality.
- **Processor**: CPU performance.

Each record is labeled with a price range category.

## Preprocessing

Preprocessing involves:
1. **Handling Missing Values**: Filling or removing missing data points.
2. **Normalization**: Scaling features for consistency across models.
3. **Feature Encoding**: Converting categorical variables into numerical format.

## Training and Evaluation

- **Training**: Each model is trained using the training dataset, with hyperparameters optimized for performance.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score are used to measure model effectiveness.

## Results

The models achieved varying degrees of accuracy, with Random Forest and SVM yielding the highest classification accuracy. Detailed metrics provide insights into the strengths of each model.

