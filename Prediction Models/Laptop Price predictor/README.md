# Laptop-price-predictor
## Table of contents
* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Data Set](#data-set)
* [File Description](#file-description)
* [Usage](#usage)
## Introduction
* This project is designed to predict laptop prices using supervised machine learning techniques. The study employs a voting regression technique as the predictive model, achieving a precision in price estimation.

* In this approach, the voting regression model combines prediction of several different models that utilizes several independent variables to predict a single dependent variable: the laptop price. The model compares actual and predicted values to assess the accuracy of its predictions.

* The project proposes a method where the price of a laptop, the dependent variable, is forecasted based on factors such as the laptop Company,Type,Inches,Screen Resoution, RAM size, weight,storage type (HDD/SSD), GPU Brand, CPU Brand, IPS ,Operating System and whether it includes a touch screen.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9676b53-fc3b-437f-9456-f52f4891a4b5" alt="Example Image" width="400"/>
</p>


## Problem Statement
<p align="center">
  <img src="https://github.com/user-attachments/assets/8e4f5d0c-ec78-43a6-bb0c-65e6a8a4bad8" alt="Alt Text" width="750" />
</p>

## Data Set
The dataset utilized in this project is 'laptop_data.csv', which includes details about laptop's different features such as brand, screen size, processor, RAM, storage, and price. It comprises 1303 rows and 12 columns. The dataset underwent significant data preprocessing, feature engineering,Exploratory Data Analysis (EDA) for analysis and for machine learning it utilizes random decision forest as the predictive model. 

## File Description
**Laptop_price_prediction.ipynb** - This Jupyter notebook contains the complete code for this project including data preprocessing, EDA, feature engineering, model building, and evaluation.

**df.pkl** - Pickle file of the processed data used in the model.

**pipe.pkl** - Pickle file of the machine learning pipeline used in the model.

**laptop_data.csv** - The dataset used in this project.

**requirements.txt** is a file used to list dependencies for the project.


## Usage
To use the laptop price predictor, follow these instructions:

1.Input the specifications of the laptop for which you want to estimate the price.

2.The system will utilize the trained model to calculate and display the predicted price.
