# Breast Cancer Classification using Naive Bayes Classifier

This project implements a **Naive Bayes Classifier** to classify breast cancer tumors as either benign or malignant using features from a breast cancer dataset.

## What is Bayesian Classification?

Bayesian classification is based on **Bayes' Theorem**, a fundamental concept in probability theory. It provides a framework for reasoning about probabilities, and in the context of machine learning, it allows us to predict the probability of a label given the features of the data.

### Bayes' Theorem

Bayes' Theorem is mathematically represented as:

P(A|B) = ((P(B|A).P(A))/(P(B)

Where:
- **P(A|B)** is the probability of event **A** occurring given that **B** is true (posterior probability).
- **P(B|A)** is the probability of event **B** given that **A** is true (likelihood).
- **P(A)** is the probability of event **A** (prior probability).
- **P(B)** is the probability of event **B** (marginal likelihood).

In the case of classification, **A** is the class (e.g., benign or malignant), and **B** is the set of features describing a data point (e.g., measurements of the tumor).

### Naive Bayes Classifier

The **Naive Bayes Classifier** is a type of Bayesian classifier that makes a simplifying assumption: it assumes that the features (variables) are conditionally independent given the class. This assumption is often called "naive," but it still works well in many real-world problems.

In the Naive Bayes Classifier, we calculate the posterior probability for each class and assign the class label with the highest probability.

### Why Use Naive Bayes?

- **Simplicity**: Naive Bayes is easy to implement and requires a small amount of training data to estimate the parameters (mean and variance of the features).
- **Efficiency**: It is computationally efficient and can handle large datasets.
- **Performance**: Despite its simplicity, Naive Bayes often performs surprisingly well, especially for text classification tasks (e.g., spam detection) and binary classification problems like the breast cancer classification task in this project.


## Dataset

The dataset used is the **Breast Cancer Wisconsin Dataset**, which contains various features related to cell nuclei present in images of breast tissue. The task is to predict whether a tumor is benign (B) or malignant (M) based on these features.
`https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset`

### Features:
- Radius, texture, perimeter, area, and smoothness (mean, worst, and standard error values)
- Compactness, concavity, symmetry, and more…

### Target:
- **Diagnosis**:
  - B = Benign (Non-cancerous)
  - M = Malignant (Cancerous)


## Libraries Used

This project uses the following Python libraries:

- **Pandas**: For data loading and manipulation.
- **Numpy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms, preprocessing, and evaluation metrics.
- **Seaborn**: For visualizing the confusion matrix.
- **Matplotlib**: For plotting graphs (e.g., ROC curve).

## Steps Involved

1. **Data Loading**: The dataset is loaded using pandas and inspected for structure.
2. **Preprocessing**: Features are scaled using `StandardScaler` and the target variable (diagnosis) is encoded into numerical labels using `LabelEncoder`.
3. **Model Training**: A **Gaussian Naive Bayes** classifier is trained on the preprocessed dataset.
4. **Evaluation**: The model is evaluated using:
   - Confusion matrix
   - Classification report (precision, recall, F1-score)
   - ROC curve (to visualize the model's true positive rate vs. false positive rate).


