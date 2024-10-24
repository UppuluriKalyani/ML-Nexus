import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK data if not already done
nltk.download('stopwords')

# Step 1: Load the Dataset
# You can download the "SMS Spam Collection" dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
# Assuming you have the data as a CSV file named 'SMSSpamCollection.csv' with columns 'label' and 'message'

data = pd.read_csv('SMSSpamCollection.csv', sep='\t', names=['label', 'message'])

# Step 2: Data Preprocessing
# Convert labels to binary values: 'spam' to 1 and 'ham' to 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Clean the text data
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['message'] = data['message'].apply(preprocess_text)

# Step 3: Split the Data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Extraction
# Use TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
