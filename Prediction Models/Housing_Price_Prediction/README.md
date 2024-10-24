# Housing Price Prediction Model

## Table of Contents
* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Data Set](#data-set)
* [File Description](#file-description)
* [Usage](#usage)

## Introduction
This project is designed to predict housing prices using supervised machine learning techniques. The study employs a regression model to achieve accurate price estimations.

In this approach, the regression model uses various independent variables to predict a single dependent variable: the housing price. The model compares actual and predicted values to assess the accuracy of its predictions.

The project forecasts housing prices based on factors such as location, square footage, number of bedrooms, number of bathrooms, year built, and additional features like garages and gardens.

![Screenshot 2021-10-20 023352](https://user-images.githubusercontent.com/49127037/138058027-a9d85497-8bbb-4385-b313-1b0dca7e3dbb.png)

## Problem Statement
To understand and analyze Real Estate features and their effect on market price, **Real Estate Investors can use this model to take advantage of opportunities to buy, sell, renovate in the extremely profitable and money-making market**. Without a model, often vague estimates are done which hurts the revenue badly. This machine learning model takes into different factors like location of the house, grade of the house, influential factors on determining the price of the house, analyzes all these factors deeply through visualization and statistical models, and thereby solves a huge problem of the Real Estate Investors.


## Data Set
The dataset utilized in this project is 'housing_data.csv', which includes details about various features related to housing such as location, size, number of rooms, and price. It comprises 1500 rows and 10 columns. The dataset underwent significant data preprocessing, feature engineering, and Exploratory Data Analysis (EDA) to prepare it for machine learning, utilizing regression algorithms as predictive models.

## File Description
**Housing_Price_Prediction.ipynb** - This Jupyter notebook contains the complete code for this project, including data preprocessing, EDA, feature engineering, model building, and evaluation.

**df.pkl** - Pickle file of the processed data used in the model.

**pipe.pkl** - Pickle file of the machine learning pipeline used in the model.

**housing_data.csv** - The dataset used in this project.

**requirements.txt** - A file used to list dependencies for the project.

## Usage
To use the housing price predictor, follow these instructions:

1. Input the specifications of the property for which you want to estimate the price.

2. The system will utilize the trained model to calculate and display the predicted price.
## Data Collection

The dataset is retrieved from Kaggle. It contains 20K+ house records with 18 features such as no. of bedrooms, area of of the living area, zipcodes to where the houses are,etc, and their sale price for King County. The houses are sold between May 2014 and May 2015. The machine learning model should be able to predict future house price in this county.

## Exploratory Data Analysis

Here in this [notebook](https://github.com/ishita-sadhukhan/house-price-prediction/blob/main/code%20notebooks/1.%20Data%20exploration%20and%20Data%20Cleaning.ipynb) I do basic exploratory data analysis on the dataset to get an understanding of the data. *Python packages like matplotlib and seaborn are used*. Things I covered :
* Getting an understanding of the data 
* Checking missing values and treating them for fitting any linear model
* Trimming the dataset to mid-level houses categories to suit the necessity of the real estate investors
* Analysis of the features with respect to price of the house
* Visualization of the location of the house with respect to few features of the house and how then can derive better business decision for the investor

### Data Findings and Recommendations

**1. Presence of a Waterfront** : Exploratory data analysis shows that average price of a house is significatly high when the house has a waterfront. This can be clearly seen in the next pic , that houses near waterbodies(blue dots) like Lake Washington and few others have price > $800K. This can be an important information to real estate investors when they buy or acquire a house by helping them to formulate the price of the house
![image](https://user-images.githubusercontent.com/49127037/134278694-c53d0275-f79e-42f3-b304-7d7c0d316af3.png)
![image](https://user-images.githubusercontent.com/49127037/134275956-df336e54-80fc-4926-9e4e-bce6116870d1.png)

**2. Condition of the house**: EDA also shows, better is the condition of the house, more is the average sale price. This can help the investors in planning their funds in bettering the condition of a house in terms of wear and tear. 
However, the price range of condition 3-5 is not so different, so investors can plan to spend money to better the condition of the house to atleast 3.
![image](https://user-images.githubusercontent.com/49127037/134394179-d54954cf-13aa-401c-9a3d-7f1ba4cdc817.png)

**3. Grade of the house** : The highest grade quality (9 and above) of houses are found near Bellevue and Redmond . Average and above average i.e, Grade 6 and above are found all over King County, which is great for the investors. 
Also,as we see on the next picture, that grade of a house do impact the house price, and so low grade means investors spending money out of their pocket

![image](https://user-images.githubusercontent.com/49127037/134280378-14772d32-0b03-438f-8125-1048a0bd5f9a.png)
![image](https://user-images.githubusercontent.com/49127037/134394347-02d816d5-c8a0-487b-a66c-cc59aba16a37.png)

**4. Renovation of a house** : Renovating a house can definitely bring revenue for an investor. Though the percentage of house being renovated with the current sample is less than 5% but investors can definitely use this information if they invest in a not-so-great- building 
![image](https://user-images.githubusercontent.com/49127037/134281387-39accbd2-9826-4c53-8cb7-f9a4aba33eea.png)

**5. Age of the building**: : As we see in the graph, the age of the building does not influence price. This can be because old buildings can be repaired, remodelled and  renovated,  and we have seen from above analysis that these kind of changes fetch higher sale price. This can be good news to the investor as they can always acquire an old house and then rely on beautifying the house. 
![image](https://user-images.githubusercontent.com/49127037/134281034-66b56673-437f-4015-848b-e1e2a4604814.png)

**6. House with a basement** : A house with a basement has higher average saleprice. To maximize revenue, Real Estate Investors can plan to buy majority of houses with a basement. Also, close to 40% of the houses in the sample have house with basement, so investors have a good chance in aquiring more houses with a basement
![image](https://user-images.githubusercontent.com/49127037/134282096-a71edda8-1f89-42cb-ac69-7998fe30c9db.png)

**7. Zipcode groups** : Mid-price houses are present across all of Kingcounty and Seattle. The zipcode information can help investors in finding out, which regions have the most expensive mid-price houses and so they can keep an eye whenever a new house becomes available on the market from those zipcodes
![image](https://user-images.githubusercontent.com/49127037/134394611-3a1bdebb-9e07-49a3-a83f-cf3e3c5a057d.png)
![image](https://user-images.githubusercontent.com/49127037/134283375-7e84f648-a4c8-4ff1-ba40-3744c5599428.png)

## Feature Engineering and Data Preprocessing

In this [notebook](https://github.com/ishita-sadhukhan/house-price-prediction/blob/main/code%20notebooks/2.Feature%20Engineering%20and%20Data%20Preprocessing.ipynb) I have tried following things :
* Created few features with the help of existing features , so as to derive more information to feed into the model . I have also created few graphs for data visualization to see how this new features are defining the price of the house

-Creating a variable where a house has both a waterfront and a view
-Age of the building - how is is house price related to the age of the building
-Whether the house has been renovated : whether house price gets impacted for ever being renovated or not?
-If the house has a basement : whether house price gets impacted with having a basement or not?
-Size of an average room in sq ft : how is is house price related to the average size of a room

* Created dummy variables for the categorical features. 
* A heatmap of the correlation between price and other features is produced to understand which features are highly correlated with price

### Features used for the model
The important features which are used in building models are # of bedrooms, area of the living room, location of the homes in the County, age of the buildings etc. As we have seen from the data exploration, these features have significant importance in terms of pricing of a house and thus can contribute in predicting future house price.

## Training models and evaluation and hyperparameter- tuning
Built models and then evaluate them in this [notebook](https://github.com/ishita-sadhukhan/house-price-prediction/blob/main/code%20notebooks/3.Model%20Building%20and%20Evaluation.ipynb). Steps I have done are :
* Dividing the data into two parts: training and validation. To find the model, I use the training set and finally test the model on unseen data, which is the validation set
* Creating a baseline model using the median of house price across the zipcode groups. Mean Absolute Error (MAE) is calculated as the evaluation metric which is the error. Other complex models will be tested against this baseline model MAE. 
* 4 complex models are run for this regression task and best out of the lot with the lowest MAE is chosen as the final model. I got Random Forest Regression model with the lowest MAE
* Hyperparameter tuning of the RandomForest model and finally testing on the unseen data - n_estimators, max_features, max_depth, min_samples_leaf
* Getting the feature importance of the model which shows, which features have contributed the most in the final model


### Model Findings

**1. Baseline model** : This is the most simple prediction of a house price. When no ML model is in place, the price of a house is the median price of all the houses in that zipcode group to which this house belongs. This is measured with the evaluation metric Mean Absolute Error (MAE) .
The MAE for the baseline model= 147222. This says that on average the prediction of the house is off by $147K

**2. Final Model :** : Random Forest came out as a clear winner compared to Linear Regression model and two other complex models : Decision Tree and Gradient Boosting

Models | Mean Absolute Error
------------ | -------------
Random Forest | 52,940
Gradient Boosting | 57,830
Decision TreeRegressor | 75,808
LinearRegression | 75,810

After hypertuning the Random Forest and fitting with the best estimator, 

**Final Model MAE = 51,207**

**Explained Variance by the model = 86%**

It means the final RF model's prediction is just a $51K away  from the actual value of the house and the model explains 86% of the variablity of price across the sample.
Also, from the below graph we see that there’s a strong correlation between the model’s predictions and its actual results. For a good fit, the points should be close to the fitted line
![image](https://user-images.githubusercontent.com/49127037/134292032-46ea40e6-32f3-45cd-92e6-ba865e53dba1.png)


**3. Feature Importance** - No machine learning algorithm is complete without knowing which features have the most contribution in the model. 
We see that features like latitude and longitude i.e the location of the house,the size of the interior living area, grade and condition of the house,  and average size of a room are important features to look at while predicting the house price at King County.
Real Estate investors can pay special attention to these features and see how these features are correlated with price, and then take their business decisions so as to where to invest and where to spend money
![image](https://user-images.githubusercontent.com/49127037/134293434-16b5ee9d-95f2-4d03-b191-44385b527ba6.png)

## Conclusion

Together, location of the house , square footage of the living area of the houses, grade, and the size of neighbors’ homes result as the best predictors of a house’s price in King County. The model can significantly help Real Estate investors to take these factors all together into account  while acquring a house. The model though has a limitation which is , houses those are less than a million dollar typically with 2-6 bedrooms are within the scope of this model. Some further investigations on this data showed that due to limited data on high-priced-valued homes, the machine learning model produces high residual errors. 

## Next Steps
Four supervised machine learning models are being tested on this data. As next steps, we can try to apply deep learning models such as Neural Network models so as to curb our error rate more. Also, due to limited data on high priced value homes, the scope of the model is limited to predicting <$1 million homes. Acquiring more data will help us overcome this constraint and thus can be applied to estimate expensive high-priced houses.   

## Your Signature
Pratik Kumar Panda
