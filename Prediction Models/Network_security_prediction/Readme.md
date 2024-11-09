# Network Security Prediction Models

## Project Overview

This repository contains a machine learning project aimed at improving network security by predicting the safety of incoming requests. Using a labeled dataset with request details, the project applies various machine learning models to classify requests as "safe" or "unsafe" based on request content, including potential indicators of OWASP Top 10 threats.

## Problem Statement

In the rapidly evolving field of cybersecurity, traditional methods of threat detection often struggle to adapt to new attack patterns. This project addresses the need for an adaptable, intelligent system capable of identifying potentially harmful requests in real-time. By analyzing request payloads and applying machine learning techniques, we aim to develop a model that accurately identifies unsafe requests, thereby enabling proactive defense against malicious attacks.

## Project Structure

- `data/`: Contains datasets used for training and testing the models.
- `notebooks/`: Jupyter notebooks with exploratory data analysis, feature engineering, and model building.
- `src/`: Source code for data processing, model training, and evaluation.
- `results/`: Stores the performance metrics of various models.

## Models and Techniques

1. **Data Preprocessing**: Cleaning and processing request payload data, including handling missing values and normalizing features.
2. **Feature Engineering**: Extracting key features from request payloads for input into machine learning models.
3. **Model Selection**:
   - **Bag of Words**: A simple but effective NLP technique that transforms text data into numerical features.
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Highlights important terms by down-weighting commonly occurring ones.
   - **Deep Learning (LSTM, CNN)**: Used for sequential analysis of request payloads, capturing more complex patterns in the data.

## Results

- The models achieved a test accuracy of up to **84.5%** on detecting malicious requests, with the deep learning model performing the best.
- A confusion matrix and accuracy plots are provided to illustrate model performance and areas for improvement.

