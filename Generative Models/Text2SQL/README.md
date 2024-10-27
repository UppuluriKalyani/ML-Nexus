# Natural Language to SQL Query Conversion (Text2SQL)

## Introduction
Text-to-SQL is a task in natural language processing (NLP) where the goal is to automatically generate SQL queries from natural language text. The goal is to enable non-technical users to interact with databases using plain English, simplifying complex data retrieval processes without needing SQL knowledge. This system can be particularly useful for business analysts, data teams, and anyone who needs quick access to insights from databases but lacks SQL expertise. A friendly, interactive SQL generation chatbot built with Streamlit and Hugging Face's Defog SQLCoder model is deployed for this purpose. This project allows users to upload a PDF containing database table definitions, ask SQL-related questions, and receive dynamically generated SQL queries based on the uploaded schema. The chatbot is designed to adapt to various database schemas, making it applicable across multiple industries like finance, healthcare, and retail.

## Tech stack
1. Pytorch - for CUDA GPU 
2. NLP Tools: SpaCy, Transformers
3. Machine Learning Models: Sequence-to-Sequence models, Transformers
4. SQL Query LLM : Hugging Face Defog's SQLCoder for query generation and execution
5. Database Management Systems (DBMS): MySQL
6. Frameworks: Streamlit for frontend interactive UI (alternatively select Flask for backend)

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
3. Alternatively set up a Groq account and use Groq API or OpenAI's GPT-4-turbo for dynamic generation of table definitions given a context
   ```
   groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
   ```
4. To generate a modified version of Text2SQL chatbot import the text2sql.core package that will dynamically interpret natural language queries and convert them into SQL commands for PostgreSQL, utilizing OpenAI's GPT-3.5-turbo for language understanding. Additionally, it will have functionalities to interact with PostgreSQL through the database_connector module. This implementation does not use GPU or CUDA and is CPU-Compatible.
   
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
![image](https://github.com/user-attachments/assets/d5234239-c985-4362-a8ab-0fba13775884)

