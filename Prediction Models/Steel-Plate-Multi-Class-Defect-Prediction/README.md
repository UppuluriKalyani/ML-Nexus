
# Steel Plate Defect Prediction - Kaggle Playground Series (Season 4, Episode 3)
Welcome to the repository for the 2024 Kaggle Playground Series - Season 4, Episode 3 competition on steel plate defect prediction!

## Introduction

This repository contains code and resources for participating in the Kaggle competition aimed at predicting the probability of various defects on steel plates. The competition provides an opportunity for participants to practice their machine learning skills and explore techniques for image classification and defect detection.

## **Objective:-**

Develop a machine learning model to accurately predict the probability of various defects in steel plates.

## **Evaluation:-**

- Submissions are evaluated using the ***area under the ROC curve (AUC)***, averaged across seven distinct defect categories.

## Dataset:-

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Steel Plates Faults dataset from UCI. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## **Files:-**
- **train.csv**:- the training dataset; there are 7 binary targets: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults
- **test.csv**:- the test dataset; your objective is to predict the probability of each of the 7 binary targets
- **sample_submission.csv**:- a sample submission file in the correct format
The dataset consists of steel plate images, each labeled with one or more defect categories. Participants are tasked with developing machine learning models capable of accurately predicting the probability of these defects.


    | Variable       | Description                                       |
    |----------------|---------------------------------------------------|
    | id             | Unique identifier for each steel plate image.     |
    | Pastry         | Indicates the presence of a pastry defect.        |
    | Z_Scratch      | Indicates the presence of a Z-scratch defect.     |
    | K_Scatch       | Indicates the presence of a K-scratch defect.     |
    | Stains         | Indicates the presence of stains on the plate.    |
    | Dirtiness      | Indicates the level of dirtiness on the plate.    |
    | Bumps          | Indicates the presence of bumps on the plate.     |
    | Other_Faults   | Indicates the presence of other types of faults.  |


These variables describe the defects and characteristics present in the steel plates. The presence or absence of each defect is indicated by binary values, with 1 representing the presence of the defect and 0 representing its absence.

--- 

## Installation
To install and use this repository, follow these steps:-

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yashksaini-coder/Steel-Plate-Multi-Class-Defect-Prediction
    ```

2. Navigate to the repository directory:
    ```bash
    cd steel-plate-defect-prediction
    ```

3. Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once you have cloned the repository and installed the dependencies, you can proceed with the following steps:

1. Explore the dataset:
    - Review the provided CSV files (`train.csv` and `test.csv`) to understand the data structure and features.
    - Check the sample submission file (`sample_submission.csv`) to understand the expected format for predictions.

2. Develop your machine learning models:
    - Use your preferred tools and libraries (e.g., Scikit-learn, TensorFlow, PyTorch) to develop machine learning models for predicting defect probabilities.
    - You may explore different algorithms, feature engineering techniques, and model architectures to improve prediction accuracy.

3. Submit your predictions:
    - Generate predictions for the test dataset and format them according to the guidelines provided in the competition.
    - Submit your predictions through the Kaggle competition platform before the submission deadline.

## Dependencies

- Python 3.11
- Data manipulation libraries (e.g., Pandas, NumPy)
- Machine learning libraries (e.g., Scikit-learn, TensorFlow, PyTorch)
- Jupyter Notebook (optional for experimenting and model development)

---
# **Author**:- [Yash Kumar Saini](https://www.linkedin.com/in/yashksaini/)<br>
# [Medium](https://medium.com/@yashksaini)<br>
