# **Spam Detection (Text Classification)**
### Objective
The goal of this model is to classify text data from three different languages (English, French, and German) as either Spam or Ham (not spam).
### Explanation
- We will use the given dataset to train our model.
- Next, we need to process this data, followed by data preprocessing, which involves checking for missing values and removing them where necessary.
- The data is split into training and testing sets. The training data is used to train the model, and the testing data is used to make predictions.
- We need to convert our text data into numerical representations using TfidfVectorizer.
- Then, we will initialize the Logistic Regression model. The training data is used to train this Logistic Regression model, which is preferred for binary classification tasks.
- Once this is done, we will have a trained Logistic Regression model. Now, when we input a new email into the trained model, it will predict whether the email is spam or ham (not spam).