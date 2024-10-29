import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bs4 import BeautifulSoup
import numpy as np
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import time
import tweepy
from tweepy import OAuthHandler
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

# Twitter API credentials
consumer_key = '12345'
consumer_secret = '12345'
access_token = '12345-12345'
access_secret = '12345'

# Set up Tweepy authentication
def authenticate_twitter():
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return tweepy.API(auth)

# Preprocessing functions
def clean_tweet(tweet):
    """Remove URLs and special characters from the tweet."""
    URLless_string = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    return re.sub(r'\@\w+|\#', '', URLless_string).strip()

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

# Sentiment analysis
def analyze_sentiment(documents):
    """Perform sentiment analysis on the provided documents."""
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for sentence in documents:
        sentiment = sia.polarity_scores(sentence)
        sentiments.append(sentiment['compound'])  # Store only the compound score
    return sentiments

# Collect tweets
def collect_tweets(api, term, number_tweets=30):
    """Collect tweets based on a search term."""
    tweets_data = []
    for status in tweepy.Cursor(api.search, q=term, lang='en').items(number_tweets):
        cleaned_tweet = clean_tweet(status.text)
        if cleaned_tweet:  # Only append non-empty tweets
            tweets_data.append(cleaned_tweet)
    return tweets_data

# Prepare data for topic modeling
def prepare_corpus(tweets):
    """Prepare corpus for topic modeling."""
    cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]
    texts = [tokenize_and_lemmatize(tweet) for tweet in cleaned_tweets]
    
    # Remove stopwords and words that appear only once
    stoplist = set('for a of the and to in is he she on i will it its us as that at who be'.split())
    texts = [[word for word in text if word not in stoplist] for text in texts]
    
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    return texts

# Visualize sentiment
def plot_sentiment(ax, sentiments, term):
    """Plot the cumulative sentiment over time."""
    x = np.arange(len(sentiments))
    cumulative_sentiment = np.cumsum(sentiments)
    
    ax.plot(x, cumulative_sentiment, linewidth=3, color='blue')
    ax.fill_between(x, cumulative_sentiment, 0, where=cumulative_sentiment < 0, facecolor='red', alpha=.7)
    ax.fill_between(x, cumulative_sentiment, 0, where=cumulative_sentiment > 0, facecolor='lawngreen', alpha=.7)
    ax.set_title(f"REAL-TIME Analysis of Mood in Twitter\nKeyword: {term}", fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mood')

# Main function to run the sentiment analysis
def main():
    term = 'trump'
    number_tweets = 30
    t = 0
    sentiments = []

    api = authenticate_twitter()
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        nonlocal t
        if t < 50:
            tweets = collect_tweets(api, term, number_tweets)
            corpus = prepare_corpus(tweets)
            if corpus:
                sentiments.extend(analyze_sentiment(corpus))
                plot_sentiment(ax, sentiments, term)
                t += 1
                time.sleep(1)

    ani = FuncAnimation(fig, update, frames=range(50), repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
