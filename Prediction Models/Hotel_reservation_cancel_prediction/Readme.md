# Hotel Reservation Cancellation Prediction

## Project Overview

This project develops a machine learning model to predict the likelihood of hotel reservation cancellations. By leveraging historical booking data, this model assists hotels in optimizing room occupancy and reducing revenue loss by accurately predicting cancellations.

## Problem Statement

Predicting cancellations is critical for hotels to manage their inventory, staff, and resources efficiently. This project addresses this need by creating a model that can help hotels anticipate cancellations, thereby reducing overbooking risks and maximizing occupancy rates.

## Data

The dataset includes various features related to hotel bookings, such as:
- Number of adults and children
- Number of weekend and week nights
- Meal plan types
- Required car parking space
- Room type reserved
- Lead time and other features relevant to booking behavior

## Models Used

The following machine learning models were evaluated:
1. **Linear Models**: Simple and interpretable, providing a baseline.
2. **Non-Linear Models**: Capture complex patterns in the data.
3. **Ensemble Methods**: Improve accuracy by combining multiple algorithms, including:
   - Random Forest Classifier
   - XGBoost Classifier
   - AdaBoost Classifier

## Evaluation Metrics

Models were evaluated using:
- **Accuracy**: Overall correctness of the predictions.
- **Precision**: Correct positive predictions divided by total positive predictions.
- **Recall**: True positive predictions out of actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

## Results

The best model achieved an accuracy of **89%** with high precision and recall, indicating strong performance in predicting cancellations. The confusion matrix and ROC curves highlight the model’s effectiveness in handling the cancellation and non-cancellation classes.

## Conclusion

This model allows hotels to make data-driven decisions by identifying potential cancellations early on. By doing so, hotels can optimize their operations and minimize lost revenue from unoccupied rooms.
