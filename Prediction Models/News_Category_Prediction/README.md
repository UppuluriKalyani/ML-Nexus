# 📰 News Article Classification Streamlit App

## Overview 🎯

This Streamlit application classifies news articles into five predefined categories: **Business**, **Entertainment**, **Politics**, **Sports**, and **Technology**. It leverages advanced machine learning techniques, such as **Non-Negative Matrix Factorization (NMF)** and **Support Vector Classifier (SVC)**, to provide an intuitive interface where users can input articles and receive instant predictions.

## Table of Contents 📑

- [Overview 🎯](#overview-🎯)
- [Features ✨](#features-✨)
- [Technologies Used 🔧](#technologies-used-🔧)
- [Dataset 📊](#dataset-📊)
- [Jupyter Notebook 📘](#jupyter-notebook-📘)
- [Installation ⚙️](#installation-⚙️)
- [Usage 💻](#usage-💻)
- [Model Training 🤖](#model-training-🤖)
- [Model Evaluation 📈](#model-evaluation-📈)
- [Model Tuning 🔍](#model-tuning-🔍)
- [Predicting Test Dataset 📉](#predicting-test-dataset-📉)
- [Contributing 🤝](#contributing-🤝)

## Features ✨

- **Interactive Classification**: Input any news article to instantly predict its category.
- **Performance Visualization**: See model accuracy metrics and confusion matrices to analyze the results.
- **Multiple Model Benchmarking**: Test various algorithms and find the best-performing one.
- **Easy-to-Use Interface**: Streamlined design to make the app simple and intuitive for all users.

## Technologies Used 🔧

- [Streamlit](https://streamlit.io/) - Web application framework.
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning models and evaluation metrics.
- [pandas](https://pandas.pydata.org/) - Data manipulation.
- [numpy](https://numpy.org/) - Numerical computations.
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - Data visualization.

## Dataset 📊

The app uses a **BBC News dataset** consisting of 2,225 articles spread across five categories. This dataset is divided into 1,490 articles for training and 735 for testing. You can download the dataset or use the one provided in this repository.

## Jupyter Notebook 📘

The Jupyter Notebook `news-classification.ipynb` is used for data preprocessing, model training, and evaluation. The key steps include:

- **Data Preprocessing**: Text tokenization, stemming, and stopword removal.
- **Feature Extraction**: TF-IDF to convert text to numerical features.
- **Model Training**: Training with models like Logistic Regression, Naive Bayes, and SVC.
- **Evaluation**: Classification reports and confusion matrices.

## Installation ⚙️

To run this app locally:

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 💻

1. Run the app using Streamlit:
    ```bash
    streamlit run app.py
    ```
2. Open your browser and go to `http://localhost:8501`.
3. Input any news article and get its predicted category.

## Model Training 🤖

The app evaluates several models, including:

- **Logistic Regression**: Models relationships between features and class probabilities.
- **Naive Bayes**: Probabilistic classifier with independence assumptions.
- **Support Vector Classifier (SVC)**: Finds an optimal hyperplane for classification.

The SVC model, which performed best during cross-validation, is used in the final prediction process.

## Model Evaluation 📈

Model performance is assessed using accuracy, precision, recall, and F1-score. Confusion matrices allow you to visualize misclassifications and understand model behavior better.

## Model Tuning 🔍

We use **GridSearchCV** to tune hyperparameters and improve model performance. Analyzing the most important words helps refine the model and preprocessing steps.

## Predicting Test Dataset 📉

The best-performing model is applied to predict the test dataset, giving users insights into model accuracy with unseen data.

## Contributing 🤝

Feel free to contribute! Suggestions, bug reports, or feature requests are welcome. To contribute:

1. Fork the repo.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Added new feature"
    ```
4. Push to your branch:
    ```bash
    git push origin feature-branch
    ```

## Contributor 👥

Pratik Wayal - Feel free to connect [GitHub](https://github.com/pratikwayal01)

---