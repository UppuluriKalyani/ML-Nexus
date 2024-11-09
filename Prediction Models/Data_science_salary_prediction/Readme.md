# Data Scientist Salary Prediction

This project predicts annual salaries for data scientists based on attributes like experience level, employment type, job title, remote work ratio, and company size. The goal is to develop an accurate regression model to help data professionals and companies understand compensation trends and make informed salary-related decisions.

## Project Structure

- `data`: Contains the dataset for data scientist salary prediction.
- `notebooks`: Jupyter notebooks with exploratory data analysis (EDA), feature engineering, and model training.
- `models`: Saved models for prediction.
- `README.md`: Project overview and instructions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ds-salary-prediction.git
    cd ds-salary-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset consists of features including:
- `work_year`: Year of employment
- `experience_level`: Level of experience (e.g., Junior, Senior)
- `employment_type`: Type of employment (e.g., Full-time, Part-time)
- `job_title`: Position held by the employee
- `remote_ratio`: Percentage of work done remotely
- `company_size`: Size of the company
- `salary_in_usd`: Annual salary in USD

## Methodology

1. **Data Cleaning**: Handle missing values and encode categorical variables.
2. **Exploratory Data Analysis (EDA)**: Visualize salary distribution and analyze relationships with other features.
3. **Feature Engineering**: Transform and select features that have significant correlations with salary.
4. **Model Training**: Train regression models such as Linear Regression, Random Forest, and XGBoost to predict salary.
5. **Model Evaluation**: Evaluate models using metrics like R² and RMSE, and select the best-performing model.



## Results

The best model achieves an R² score of approximately 0.73 on the test set, indicating a strong predictive performance for salary based on the given features.
