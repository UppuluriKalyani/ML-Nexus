import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize Groq API client
client = Groq(api_key=os.getenv("YOUR_GROQ_API_KEY"))

# Function to interact with Groq API for question answering
def query_groq_api(question, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Context: {context}. Question: {question}"}
        ],
        model="llama3-8b-8192",   # Used for RAG
    )
    return chat_completion.choices[0].message.content
