# Research-topic-Prediction

### Problem Statement
Researchers have access to large online archives of scientific articles. As a consequence, finding relevant articles has become more difficult. Tagging or topic modelling provides a way to give token of identification to research articles which facilitates recommendation and search process.

Given the abstract and title for a set of research articles, predict the topics for each article included in the test set. 

Note that a research article can possibly have more than 1 topic. The research article abstracts and titles are sourced from the following 6 topics: 

1. Computer Science

2. Physics

3. Mathematics

4. Statistics

5. Quantitative Biology

6. Quantitative Finance

## Approach

### Data Preprocessing:

We clean the text data by removing punctuations, converting text to lowercase, and removing unnecessary characters. This helps standardize the text input for better performance during the model training.
Feature Extraction:

We extract features from the research abstracts and titles using text vectorization techniques. The two main methods used are CountVectorizer (which counts word occurrences) and TF-IDF (Term Frequency-Inverse Document Frequency), which assigns importance to words based on their frequency across documents.
### Model Selection:

We use a Linear Support Vector Machine (LinearSVC) with a multi-output classification approach. This allows the model to predict multiple topics for each article simultaneously.
Evaluation:

The model is evaluated using common classification metrics such as precision, recall, F1-score, and accuracy. These metrics give insight into how well the model performs across different research topics.
### Prediction:

After training, the model predicts the topics for unseen research articles, and the results are formatted for submission. Each prediction shows whether a particular article belongs to one or more of the six topics.
## Project Highlights
Multi-label Classification: Each article can be tagged with more than one topic, so the model needs to handle multiple outputs simultaneously.
Text Processing: Effective text preprocessing and vectorization are key to extracting meaningful features from the research articles’ titles and abstracts.
### Model Performance:
 The model demonstrates good performance in most categories, but there is room for improvement, particularly in addressing class imbalances for topics with fewer articles.
## Future Work
Future improvements could include addressing the data imbalance, exploring advanced machine learning models such as deep learning techniques, and refining the feature extraction process for better prediction accuracy.

This project provides a foundation for automating topic prediction in research articles, potentially enhancing search engines and recommendation systems in academic databases.