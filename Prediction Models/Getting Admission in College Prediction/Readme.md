# College Admission Prediction
This project focuses on predicting the chances of getting admission to a university based on various factors using machine learning. The model compares multiple algorithms and selects the best one to predict the probability of admission accurately.

## Problem Statement
The aim of the project is to predict the likelihood of a student getting admitted to a university based on factors such as:

GRE Score
TOEFL Score
University Rating
Statement of Purpose (SOP)
Letter of Recommendation Strength (LOR)
CGPA (Cumulative Grade Point Average)
Research Experience (0 or 1)
Using these inputs, the model predicts the probability (in percentage) of admission.

## Approach
### Data Preprocessing
Before training the models, the dataset is normalized to ensure that all features are on the same scale. This step is crucial for improving the performance of various algorithms, especially for distance-based algorithms like K-Nearest Neighbors (KNN) and linear models.

### Model Selection
Several regression algorithms are explored to find the best model for predicting the chances of admission. The models considered are:

Linear Regression
Lasso Regression
Support Vector Regression (SVR)
Decision Tree Regression
Random Forest Regression
K-Nearest Neighbors Regression (KNN)
Each of these models is trained with different hyperparameters using a grid search to identify the optimal configuration for the best performance.

### Cross-Validation
Cross-validation is used to assess the generalizability of the models. The dataset is split into multiple folds, and the model performance is evaluated across these folds to ensure the model does not overfit or underfit the data.

### Model Evaluation
The performance of each model is compared using metrics like accuracy. The best-performing model is selected based on its accuracy and cross-validation score. In this project, Linear Regression proved to be the best model for predicting college admissions.

### Training and Testing
The dataset is split into training and test sets. The best model is trained on the training data and evaluated on the test data to gauge its performance in a real-world scenario. The model's accuracy is reported based on how well it predicts admissions on unseen data.

### Prediction
Once the model is trained and tested, it can predict the probability of admission based on new input data. The inputs are provided in the form of GRE, TOEFL, CGPA, and other factors, and the model outputs a probability percentage indicating the chances of admission.

## Results
Linear Regression was identified as the best model with the highest accuracy of around 81.0% during cross-validation and 82.15% on the test data.
Other models, such as Random Forest and K-Nearest Neighbors, also performed well but were outperformed by Linear Regression.
### Key Features
Automated Model Selection: Different models are tested and compared automatically using grid search, and the best model is selected based on performance.
Cross-Validation: The project incorporates cross-validation to ensure the robustness of the model.
Prediction Capabilities: The model can predict the probability of admission to a university based on specific input values.
### How It Works
The dataset is preprocessed by normalizing the features.
Multiple regression models are trained using grid search to find the best hyperparameters.
The model with the best performance (in this case, Linear Regression) is selected.
The model is trained and tested, with its accuracy reported.
The trained model is then used to make predictions about admission chances for new student profiles.
## Conclusion
The project successfully predicts the chances of admission to universities based on key factors. The Linear Regression model was the most effective, achieving high accuracy and providing reliable predictions. This model can be useful for students and institutions to predict the likelihood of getting admitted based on various academic and extracurricular factors.