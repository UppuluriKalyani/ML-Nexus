import os
import streamlit as st
import google.generativeai as genai
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import speech_recognition as sr
from gtts import gTTS
import tempfile

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Groq API client
groq_api =  st.secrets["YOUR_GROQ_API_KEY"]
client = Groq(api_key=groq_api)
# client = Groq(api_key=os.getenv("YOUR_GROQ_API_KEY"))

# Voice input and output setup
recognizer = sr.Recognizer()

# Function to interact with Groq API for question answering
def query_groq_api(question, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Context: {context}. Question: {question}"}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# gTTS-based voice output function
def voice_output(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    audio_file = open("output.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    os.remove("output.mp3")

# Load and process multiple documents
def load_and_preprocess_documents(files):
    documents = []
    for file in files:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Load the PDF using the file path
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)

        # Remove the temp file
        os.remove(tmp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

# Create FAISS vector store from the documents
def create_faiss_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Replace with the embeddings of your choice
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Query the document using FAISS vector store
def query_document(question, vector_store):
    docs = vector_store.similarity_search(question)
    context = " ".join([doc.page_content for doc in docs[:3]])  # Retrieve top 3 relevant docs
    return context

# Voice input function using SpeechRecognition
def voice_input():
    with sr.Microphone() as source:
        st.write("Listening for your query...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            st.write(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.write("Could not understand audio")
            return None

# Streamlit Interface
st.title("SmartDocMate: AI-powered Multi-Document Assistant")

# Multi-file upload for PDF documents
uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Processing documents...")
    # Load and preprocess documents
    texts = load_and_preprocess_documents(uploaded_files)
    vector_store = create_faiss_vector_store(texts)
    st.write("Documents processed successfully.")

    # Query input options (Text or Voice)
    query_option = st.selectbox("Select input method:", ("Text", "Voice"))
    
    # For text input query
    if query_option == "Text":
        question = st.text_input("Enter your query")
        if st.button("Submit"):
            context = query_document(question, vector_store)
            answer = query_groq_api(question, context)
            st.write(f"Answer: {answer}")
            voice_output(answer)

    # For voice input query
    elif query_option == "Voice":
        if st.button("Record Query"):
            question = voice_input()
            if question:
                context = query_document(question, vector_store)
                answer = query_groq_api(question, context)
                st.write(f"Answer: {answer}")
                voice_output(answer)
