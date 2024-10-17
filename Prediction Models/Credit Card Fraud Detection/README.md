## Credit Card Fraud Detection Project ##

### Technologies Used ###
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

### Dataset ###

The dataset used for this project is a modified version of the European Credit Card Transactions dataset, which contains credit card transactions made by European cardholders. It includes features that have been transformed and normalized to protect the privacy of individuals.

### Data Features ###

* Time: Time elapsed since the first transaction in seconds.
* V1 to V28: Anonymized features generated using PCA (Principal Component Analysis).
* Amount: Transaction amount.
* Class: Target variable where 1 indicates fraud and 0 indicates a legitimate transaction.

### Installation ###

To get started with the project, download the dataset from [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud], clone the repository, install the required packages and move the csv file into the cloned folder:

```
    git clone repository_url
    cd './Prediction Models/Credit Card Fraud Detection'
    pip install -r requirements.txt
```

### Usage ###

* **Data Preparation**: Load the dataset and perform any necessary preprocessing steps (e.g., scaling, handling missing values).
* **Model Training**: Use the Random Forest algorithm to train the model on the training dataset.
* **Model Prediction**: Predict fraud on the test dataset.
* **Visualization**: Visualize the results and performance metrics.

To run the Jupyter Notebook:

```
    jupyter notebook
```

Open fraud.ipynb and follow the steps outlined in the notebook.

### Model Evaluation ###

The performance of the model can be evaluated using various metrics:

* Confusion Matrix
* Accuracy, Precision, Recall Scores
* ROC-AUC Curve

These metrics help assess how well the model is able to distinguish between fraudulent and legitimate transactions.