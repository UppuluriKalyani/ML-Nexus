# Mushroom Classification using Machine Learning

This repository contains a machine learning model to classify mushrooms as either edible or poisonous based on their features. By analyzing attributes like color, shape, and odor, the model aims to identify toxic mushrooms, providing a practical tool for foragers and researchers.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Classification Models](#classification-models)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

Mushroom poisoning can have serious health consequences. Identifying poisonous mushrooms traditionally requires expertise, making it challenging for laypersons. This project aims to develop a machine learning model that predicts whether a mushroom is poisonous or edible based on its physical characteristics, providing an accessible tool for safe foraging and ecological studies.

## Project Overview

This project applies several machine learning models to:
1. **Train on a dataset of mushroom features**: Features like cap color, gill size, and odor are used as predictors.
2. **Classify mushrooms as edible or poisonous**: Output is a binary classification indicating toxicity.

## Classification Models

The project compares multiple classification algorithms to determine the most effective:
- **Logistic Regression**: Serves as a baseline binary classifier.
- **Decision Trees**: Captures non-linear relationships among features.
- **Random Forest**: An ensemble model enhancing decision tree performance.
- **K-Nearest Neighbors (KNN)**: Classifies based on feature similarity.

## Dataset

The dataset contains mushroom attributes relevant to classification, such as:
- **Cap shape and color**
- **Gill attachment and size**
- **Odor and habitat**

Each entry is labeled as either **edible** or **poisonous**.

## Preprocessing

Preprocessing includes:
1. **Encoding**: Transforming categorical variables into numerical format.
2. **Normalization**: Scaling features for uniform model input.
3. **Splitting**: Dividing data into training, validation, and testing sets.

## Training and Evaluation

- **Training**: Each model is trained on labeled mushroom data using cross-entropy loss.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score are used to evaluate model performance on unseen data.

## Results

The models show varying degrees of accuracy, with Random Forest and Decision Trees achieving the highest accuracy in classifying mushrooms as edible or poisonous.


