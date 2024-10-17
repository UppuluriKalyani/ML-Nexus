import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Utility function for vector store creation
def create_vectorstore(text_chunks, openai_api_key):
    """
    Create a FAISS vector store from text chunks using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# API key validation utility
def valid_api_keys(openai_key, groq_key):
    """
    Validate OpenAI and Groq API keys.
    """
    return openai_key.startswith('sk-') and groq_key.startswith('gsk_')

# Load API keys from environment variables
def load_api_keys():
    """
    Load API keys from environment variables or a .env file.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not openai_api_key or not groq_api_key:
        raise ValueError("API keys are missing! Please provide valid OpenAI and Groq API keys.")
    
    return openai_api_key, groq_api_key
