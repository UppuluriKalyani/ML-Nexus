# X sentimental analysis

### Key Components and Functionality

1. **Libraries Imported:**
   - **NLTK**: Used for natural language processing tasks, such as tokenization, part-of-speech tagging, and sentiment analysis.
   - **Matplotlib**: Used for plotting data and creating visualizations.
   - **BeautifulSoup**: Typically used for web scraping, though not utilized in this specific snippet.
   - **Tweepy**: A library for accessing the Twitter API.
   - **Gensim**: For topic modeling and creating a corpus from documents.

2. **Twitter API Authentication:**
   - The code sets up authentication using `consumer_key`, `consumer_secret`, `access_token`, and `access_secret` to access the Twitter API.

3. **Data Collection:**
   - The code collects a specified number of tweets (30 in this case) containing the term "trump" using `tweepy.Cursor`.

4. **Text Processing:**
   - Tweets are processed by removing URLs and tokenizing text. The `WordNetLemmatizer` is used to lemmatize tokens, reducing words to their base forms.
   - Part-of-speech tagging is applied to identify nouns and other parts of speech.
   - A frequency distribution of long words (between 4 and 10 characters) is created.

5. **Topic Modeling:**
   - The code creates a corpus for topic modeling using Gensim's `Dictionary` and `Tf-idf` model.
   - Latent Semantic Indexing (LSI) is applied to identify topics within the tweets.

6. **Clustering:**
   - The code uses K-Means clustering to identify clusters within the tweet data based on the processed text.

7. **Sentiment Analysis:**
   - It employs the VADER sentiment analyzer to determine the sentiment polarity of each tweet (positive, negative, neutral).
   - The sentiments are accumulated over time, creating an array of sentiment scores.

8. **Real-time Visualization:**
   - The code generates a real-time plot of sentiment over time, updating every second.
   - Positive sentiment is shown in green, negative in red, and the cumulative sentiment score is plotted.

9. **Video Output:**
   - The visualization is saved as a video (`Twitter_REAL_Time_temp.mp4`) using Matplotlib's animation capabilities.

