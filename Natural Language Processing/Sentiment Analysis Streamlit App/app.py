# Save this as app.py
import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'sentiment_analysis_model.pkl' is in the same directory.")
        return None

# Function to make predictions
def predict_sentiment(text, model):
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    return prediction

# Emoji mapping for sentiments
sentiment_emojis = {
    'happiness': '😊',
    'sadness': '😢',
    'anger': '😠',
    'love': '❤️',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐',
    'worry': '😟',
    'empty': '😶',
    'enthusiasm': '🤗',
    'fun': '😄',
    'hate': '😡',
    'boredom': '😑',
    'relief': '😌'
}

def get_sentiment_emoji(sentiment):
    return sentiment_emojis.get(sentiment.lower(), '')

def main():
    # Set page config
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="🎭"
    )

    # Add custom CSS to center content
    st.markdown("""
        <style>
        .block-container {
            max-width: 700px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Center the button */
        .stButton > button {
            display: block !important;
            margin: 0 auto !important;
            width: 200px;
        }
        
        /* Remove label */
        .stTextArea label {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)

    # Main title
    st.markdown("<h1 style='text-align: center;'>🎭 Sentiment Analysis App", unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    model = load_model()

    if model:
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            height=100,
            placeholder="Type or paste your text here..."
        )

        # Add analyze button
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip():
                # Show spinner while processing
                with st.spinner("Analyzing sentiment..."):
                    # Get prediction
                    prediction = predict_sentiment(text_input, model)
                    emoji = get_sentiment_emoji(prediction)

                    # Display results
                    st.markdown("### Results")
                    st.markdown(f"**Detected Sentiment:** {prediction} {emoji}")
            else:
                st.warning("Please enter some text to analyze.")

        # Add information about the app
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses a machine learning model to analyze the sentiment of text. 
        It can detect various emotions in the text you provide.
        
        **How to use:**
        1. Enter or paste your text in the text area
        2. Click the 'Analyze Sentiment' button
        3. View the detected sentiment
        """)
        
        st.markdown("---")
        st.caption("Developed by Rakhesh Krishna")

if __name__ == "__main__":
    main()