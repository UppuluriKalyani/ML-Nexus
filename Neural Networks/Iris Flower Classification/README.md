# Iris Flower Classification

## Project Overview

This project aims to classify iris flowers into one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

Using the measurements of the sepals and petals (length and width), we employ various machine learning models to predict the species of an iris flower. The dataset used for this project is the well-known **Iris dataset**, which is often used for classification tasks in machine learning.

## Objective

The primary goal of this project is to design a machine learning model that can:
- Accurately classify iris flowers based on their sepal and petal measurements.
- Generalize well to new, unseen data to ensure robust predictions in real-world scenarios.

## Dataset Information

- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Samples per class**: 50
- **Total samples**: 150
- **Features**:
  - Sepal length (in cm)
  - Sepal width (in cm)
  - Petal length (in cm)
  - Petal width (in cm)

## Project Workflow

1. **Data Exploration**: 
   - Visualize the relationships between features using plots like pair plots and correlation matrices.
   
2. **Data Preprocessing**:
   - Standardize the data for better model performance.
   - Split the dataset into training and testing sets (e.g., 80% training, 20% testing).
   
3. **Model Selection**:
   - Train multiple classification models to compare performance:
     - Decision Tree
     - k-Nearest Neighbors (k-NN)
     - Support Vector Machine (SVM)
     - Logistic Regression
     - Naive Bayes
   
4. **Evaluation**:
   - Models are evaluated on the test set using the following metrics:
     - Accuracy
     - Root Mean Squared Error (RMSE)
     - R² Score
     
5. **Deployment**:
   - The best-performing model can be saved and used to predict the species of new iris flowers based on input measurements.

## Getting Started

### Prerequisites

Make sure to install the required libraries before running the project. You can install them using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running the Project

1. **Clone the Repository**:

```bash
git clone https://github.com/yourusername/iris-flower-classification.git
cd iris-flower-classification
```

2. **Run the Script**:

```bash
python iris_classification.py
```

This script will train the model and display performance metrics such as accuracy and RMSE.

## Model Evaluation

The performance of various machine learning models was evaluated on the Iris dataset. The following table summarizes the results of these models, including **accuracy**, **RMSE** (Root Mean Squared Error), and **R² Score**:

| Model                        | Accuracy        | RMSE   | R² Score |
|------------------------------|-----------------|--------|----------|
| **Decision Tree Classifier**  | 96%             | 0.21   | 0.93     |
| **k-Nearest Neighbors (k-NN)**| 95.56%          | 0.21   | 0.93     |
| **Support Vector Machine (SVM)**| **98%**        | 0.15   | **0.97** |
| **Logistic Regression** (Petals)| 97.78%        | 0.15   | 0.97     |
| **Naive Bayes**               | 97.78%          | 0.15   | 0.97     |

### Key Metrics:
- **Accuracy**: The proportion of correctly classified flowers.
- **RMSE**: A measure of the prediction errors. Lower values indicate better performance.
- **R² Score**: The proportion of variance in the dependent variable that is predictable from the features. Values closer to 1 indicate a better fit.

The **Support Vector Machine (SVM)** achieved the highest accuracy (98%) with the lowest RMSE (0.15) and highest R² score (0.97), making it the best-performing model.

## Visualization

Here are some example visualizations used during data exploration:

1. **Pair Plot**: A visualization showing pairwise relationships between features, colored by species.  
   ![2D Scatter Plot](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/2DScatter.png?raw=true)

2. **Data Visualization**: A plot visualizing the data distribution and the relationships between features.  
   ![Data Visualization](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/DataVisualization.png?raw=true)

3. **Decision Tree**: A visual representation of the decision tree model used in the classification process.  
   ![Decision Tree](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/DecisionTree.png?raw=true)

4. **Heatmap 1**: Correlation matrix heatmap showing feature correlations.  
   ![Heatmap](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/Heatmap.png?raw=true)

5. **Heatmap 2**: An alternative version of the heatmap with enhanced visual clarity.  
   ![Heatmap 2](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/Heatmap2.png?raw=true)

6. **k-NN Visualization**: Visual representation of how the k-Nearest Neighbors algorithm classifies iris species.  
   ![k-NN Visualization](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/KNN.png?raw=true)

7. **Length Histogram**: Distribution of sepal and petal lengths across different iris species.  
   ![Length Histogram](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/Length_Histogram.png?raw=true)

8. **SVM Classification Boundaries**: Visualization of decision boundaries for the Support Vector Machine model.  
   ![SVM](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/SVM.png?raw=true)

9. **Width Histogram**: Distribution of sepal and petal widths across different iris species.  
   ![Width Histogram](https://github.com/aviralgarg05/ML-Nexus/blob/main/Neural%20Networks/Iris%20Flower%20Classification/Width_Histogram.png?raw=true)

## Conclusion

This project demonstrates the process of building and evaluating machine learning models for classifying iris flowers. Among the models, **SVM** provided the best performance in terms of accuracy and generalization.

## Future Work

- **Hyperparameter Tuning**: Use techniques like Grid Search or Randomized Search to optimize model performance.
- **Advanced Models**: Experiment with more complex algorithms such as **Gradient Boosting** or **Neural Networks**.
- **Deployment**: Convert the model into a web API or mobile app for real-time iris classification.

## References

- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
