# Sleep Disorder Predictor

## Overview

The Sleep Disorder Predictor project aims to predict sleep disorders using a comprehensive dataset that includes various health and lifestyle parameters. The model leverages machine learning techniques to provide insights into potential sleep issues, including insomnia and sleep apnea.

## Dataset Overview

The Sleep Health and Lifestyle Dataset covers a wide range of variables related to sleep and daily habits. It includes details such as gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps, and the presence or absence of sleep disorders.

### Key Features of the Dataset

- **Comprehensive Sleep Metrics**: Explore sleep duration, quality, and factors influencing sleep patterns.
- **Lifestyle Factors**: Analyze physical activity levels, stress levels, and BMI categories.
- **Cardiovascular Health**: Examine blood pressure and heart rate measurements.
- **Sleep Disorder Analysis**: Identify the occurrence of sleep disorders such as insomnia and sleep apnea.

### Dataset Columns

- **Person ID**: An identifier for each individual.
- **Gender**: The gender of the person (Male/Female).
- **Age**: The age of the person in years.
- **Occupation**: The occupation or profession of the person.
- **Sleep Duration (hours)**: The number of hours the person sleeps per day.
- **Quality of Sleep (scale: 1-10)**: A subjective rating of the quality of sleep, ranging from 1 to 10.
- **Physical Activity Level (minutes/day)**: The number of minutes the person engages in physical activity daily.
- **Stress Level (scale: 1-10)**: A subjective rating of the stress level experienced by the person, ranging from 1 to 10.
- **BMI Category**: The BMI category of the person (e.g., Underweight, Normal, Overweight).
- **Blood Pressure (systolic/diastolic)**: The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.
- **Heart Rate (bpm)**: The resting heart rate of the person in beats per minute.
- **Daily Steps**: The number of steps the person takes per day.
- **Sleep Disorder**: The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).

### Details about Sleep Disorder Column

- **None**: The individual does not exhibit any specific sleep disorder.
- **Insomnia**: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
- **Sleep Apnea**: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.

## Project Workflow

1. **Data Preprocessing**:

   - Handling missing values.
   - Feature scaling and normalization.

2. **Exploratory Data Analysis (EDA)**:

   - Analyzing data distribution, correlations, and visualizing patterns.

3. **Data Balancing**:

   - Addressing class imbalance using techniques like SMOTE.

4. **Model Building**:

   - Training multiple models including Logistic Regression, XGBoost, Random Forest, etc.

5. **Model Evaluation**:
   - Evaluating model performance using metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `imbalanced-learn`, `xgboost`
- **Environment**: Jupyter Notebook for code execution and visualization

## Model Evaluation

Models were evaluated using multiple metrics, including:

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to all actual positives.
- **F1-Score**: Weighted average of Precision and Recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, indicating the model’s ability to distinguish between classes.

## Usage

You can use the model to predict sleep disorders by calling the `get_prediction` function with appropriate parameters. Example usage in Python:

```python
from sleep_disorder_predictor.model import get_prediction

result = get_prediction(
    Age=30,
    Sleep_Duration=7.5,
    Heart_Rate=70,
    Daily_Steps=8000,
    Systolic=120,
    Diastolic=80,
    Occupation='Engineer',
    Quality_of_Sleep=4,
    Gender='Male',
    Physical_Activity_Level=3,
    Stress_Level=2,
    BMI_Category='Normal'
)
print(result)
```

## Recommendations

To prevent sleep disorders, consider the following lifestyle changes:

- **Regular Exercise**: Engage in physical activities for at least 30 minutes most days of the week.
- **Healthy Diet**: Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.
- **Stress Management**: Practice relaxation techniques such as meditation, yoga, or deep breathing exercises.
- **Consistent Sleep Schedule**: Go to bed and wake up at the same time every day, even on weekends.
- **Limit Caffeine and Alcohol**: Reduce intake of stimulants and depressants, especially close to bedtime.
- **Create a Relaxing Bedtime Routine**: Wind down with calming activities before sleep, such as reading or taking a warm bath.
