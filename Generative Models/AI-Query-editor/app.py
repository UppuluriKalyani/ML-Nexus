import streamlit as st
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv(override=True)

api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def parse_gemini_response(response):
    sections = response.split('\n\n')
    sql_query = sections[0].strip()
    explanation = sections[1].strip() if len(sections) > 1 else ""
    optimization = sections[2].strip() if len(sections) > 2 else ""
    
    return {
        'query': sql_query,
        'explanation': explanation,
        'optimization': optimization
    }

st.set_page_config(page_title="AI SQL Query Generator", layout="centered")
st.title("🧠 AI-Powered SQL Query Generator")
st.write("Transform natural language into optimized SQL queries using Google’s Gemini Pro AI.")
st.markdown("### Provide the Database Schema")
st.write("Paste your database schema below (e.g., CREATE TABLE statements).")
schema_input = st.text_area("Database Schema", height=200, placeholder="Example:\nCREATE TABLE users (\n    id INT PRIMARY KEY,\n    name VARCHAR(100),\n    email VARCHAR(100)\n);")
st.markdown("### Describe Your Query in Plain Language")
st.write("Explain the data request you want (e.g., 'Show me all users who signed up last month').")
query_input = st.text_area("Your Query", height=100, placeholder="Example: Show me all users who signed up last month")
col1, col2 = st.columns([1, 3])
with col1:
    generate_button = st.button("Generate SQL Query")

with col2:
    clear_button = st.button("Clear Input")

if generate_button:
    if not schema_input or not query_input:
        st.error("Please provide both schema and query description")
    else:
        try:
            with st.spinner("Generating SQL query..."):
                full_prompt = f"""
                Based on the provided database schema:
                {schema_input}

                Formulate an optimized SQL query to satisfy the following request:
                {query_input}

                Deliver the following:
                1. The SQL query in its final form.
                2. A detailed explanation of the query’s logic and how it satisfies the request.
                3. Suggestions for optimizing the query for better performance, considering indexing, query structure, and database-specific features.
                """
                response = model.generate_content(full_prompt)
                result = parse_gemini_response(response.text)

                st.success("SQL query generated successfully!")
                st.subheader("Generated SQL Query")
                st.code(result['query'], language='sql')
                st.subheader("Explanation")
                st.write(result['explanation'])
                st.subheader("Optimization Suggestions")
                st.write(result['optimization'])

        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            st.error("An error occurred while generating the SQL query. Please try again.")

if clear_button:
    st.experimental_rerun()
