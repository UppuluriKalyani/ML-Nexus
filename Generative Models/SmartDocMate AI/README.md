# SmartDocMate - AI-Powered Interactive Document Assistant

An AI-powered interactive assistant that can deeply understand and help users navigate complex documents such as legal contracts, research papers, technical manuals, or lengthy reports. Using GenAI, AI agents, and vector databases, the system intelligently summarizes sections, answers specific questions, and retrieves relevant content based on the user’s query. This system also incorporates generation of voice output from text using gTTS and Streamlit's st.audio for playback.  Additionally, voice to text can also be performed using PyAudio. 

## How It Works:

1. Document Ingestion: Users upload documents (PDF, Word, etc.).
2. Content Summarization: Using large language models (LLMs) powered by LangChain, the system provides concise, readable summaries of key sections.
3. Contextual Q&A: Users can ask specific questions about the document. For example, "What are the terms of termination in this contract?" or "What does the report say?"
4. Smart Navigation: Based on the question, the system provides relevant excerpts and explains them in plain language using the Groq API for inference.
5. Memory & Search: Using vector databases like FAISS, the system stores embeddings of document sections to enable quick retrieval for similar queries in the future, allowing faster responses to common questions.

## Technology Stack:
 - Groq API
 - LangChain
 - AI Agents
 - Vector Databases (Pinecone/FAISS)
 - LLMs (OpenAI or Google generativeAI)
 - gTTS
 - speech_recognition 
 - PyAudio
 - Streamlit : A user-friendly interface where users can upload documents, ask questions, and get summaries.

## Installation requirements

1. Fork the repository and install all the necessary modules from requirements.txt file
2. Set up your free Groq API account and create an API key. Ensure to replace "your_groq_api_key" in the code with your created key.
3. Import the standard Chat Completion template from Groq and use model="llama3-8b-8192" or any available chat completion models as a compulsory parameter for language model inference.
4. Use GoogleGenerativeAIEmbeddings from Langchain with the correct model identifier .Ensure your Google GenAI API key is set up properly in your environment.
5. Additionaly integrate **gTTS** (Google Text-to-Speech) for generating voice output text-to-speech and Streamlit's **st.audio** for playback.
6. Run the streamlit app
   <code>
                                            streamlit run app.py
   </code>

## Screenshots
![Screenshot_1](https://github.com/user-attachments/assets/18c004e3-9275-45d4-8f42-ff572383221b)
![Screenshot_2](https://github.com/user-attachments/assets/433b46a5-ae2f-4805-825c-babac5a98246)

