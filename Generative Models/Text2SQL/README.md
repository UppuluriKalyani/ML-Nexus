# Natural Language to SQL Query Conversion (Text2SQL)

## Introduction
Text-to-SQL is a task in natural language processing (NLP) where the goal is to automatically generate SQL queries from natural language text. The goal is to enable non-technical users to interact with databases using plain English, simplifying complex data retrieval processes without needing SQL knowledge. This system can be particularly useful for business analysts, data teams, and anyone who needs quick access to insights from databases but lacks SQL expertise. A friendly, interactive SQL generation chatbot built with Streamlit and Hugging Face's Defog SQLCoder model is deployed for this purpose. This project allows users to upload a PDF containing database table definitions, ask SQL-related questions, and receive dynamically generated SQL queries based on the uploaded schema. The chatbot is designed to adapt to various database schemas, making it applicable across multiple industries like finance, healthcare, and retail.

## Tech stack
1. Pytorch - for CUDA GPU 
2. NLP Tools: SpaCy, Transformers
3. Machine Learning Models: Sequence-to-Sequence models, Transformers
4. SQL Query LLM : Hugging Face Defog's SQLCoder for query generation and execution
5. Database Management Systems (DBMS): MySQL
6. Frameworks: Streamlit for frontend interactive UI, Flask for backend

## Features
 - Chatbot Interface: Interact with a conversational bot that can greet you, respond to common phrases, and answer SQL-related questions.
 - Dynamic Schema Extraction: Upload a PDF with table definitions, and the chatbot extracts and understands the schema, enabling context-aware SQL generation.
-  SQL Generation: Generate SQL queries based on user-provided questions and dynamically extracted schema from the PDF file.
- Friendly Interaction: The bot responds to common phrases like "hello," "thanks," and "bye," providing a more engaging experience.

## Prerequisites
1. Python 3.7+
2. Install required libraries:
   ```
   pip install torch transformers bitsandbytes accelerate sqlparse streamlit PyPDF2
   ```
## Running the App
1. Clone the repository
  ```
   git clone https://github.com/yourusername/Text2SQLChatbot.git
  ```
3. Navigate to the project directory
  ```
  cd Text2SQLChatbot
```
5. Start the Streamlit app
```
streamlit run app.py
```
## Screenshots
