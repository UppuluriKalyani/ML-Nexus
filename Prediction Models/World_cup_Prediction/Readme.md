# World Cup Prediction with Deep Learning

This project uses a Character Embedding model to predict FIFA World Cup match outcomes with an 80% accuracy. The prediction model is developed using TensorFlow, with a focus on leveraging player-specific data along with team information to enhance prediction accuracy.

## Project Overview

In this project, we integrate individual player data with match history to construct a deep learning model that predicts match results. This model demonstrates the importance of feature selection, showing how player data and match records contribute to accurate predictions.

### Key Features

- **Data Ingestion:** Load and preprocess historical World Cup match data and player rosters.
- **Data Transformation:** Create character embeddings for player names to use in deep learning model inputs.
- **Model Building:** Use TensorFlow to construct a Character Embedding model tailored for predictive analysis.
- **Evaluation:** Achieve 80% prediction accuracy on test data, using World Cup 2018 results as a validation set.

## Project Structure

- `data/`: Contains historical World Cup data and player rosters.
- `notebooks/`: Jupyter notebooks with data preprocessing, model building, and evaluation code.
- `scripts/`: Python scripts for data loading, model training, and evaluation.
- `models/`: Saved models for reproducibility.

## Getting Started

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the model**:
    - Load the data in the `data/` directory.
    - Execute the Jupyter notebooks in `notebooks/` to preprocess data, build, and evaluate the model.

3. **Model Evaluation**:
    - The model's performance metrics are logged, including accuracy, precision, and recall.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Results

The model achieves approximately 80% accuracy in predicting the outcome of FIFA World Cup matches. This result highlights the effectiveness of character embeddings for player names and team rosters in sports outcome prediction.

## Conclusion

The project emphasizes that feature selection and data quality are as crucial as model architecture in predictive modeling. By focusing on player-specific data, this project offers insights into the predictive power of individual player contributions to team performance.

