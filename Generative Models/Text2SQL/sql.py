import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlparse
import PyPDF2

# Load the model
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True,
)

# Set up Streamlit page
st.title("Friendly SQL Generation Chatbot")

# Chatbot greetings
st.write("👋 Hello! I'm your SQL assistant! You can greet me, ask questions, or upload a PDF with table definitions.")
st.write("Feel free to ask me SQL-related queries, and I will do my best to help you!")

# Handle user input
user_input = st.text_input("Your message: ")

# Bot greeting response
if "hello" in user_input.lower() or "hi" in user_input.lower():
    st.write("🤖: Hi there! How can I assist you with SQL queries today?")

# PDF upload section for table schema
uploaded_file = st.file_uploader("Upload a PDF with your table definitions", type="pdf")

# Function to extract table schema from PDF
def extract_table_schema_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    schema_text = ""
    for page in pdf_reader.pages:
        schema_text += page.extract_text()
    return schema_text

# Display extracted table schema
if uploaded_file is not None:
    st.write("📄 Extracted Table Schema from PDF:")
    table_schema = extract_table_schema_from_pdf(uploaded_file)
    st.text_area("Extracted Schema", table_schema, height=200)

# Function to generate SQL query
def generate_query(question, schema):
    prompt = f"""### Task
Generate a SQL query to answer the following question: {question}

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'
- Remember that profit is revenue minus cost
- Remember that revenue is sale_price multiplied by quantity_sold
- Remember that cost is purchase_price multiplied by quantity_sold
### Database Schema
This query will run on the following database schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers the question:
[SQL]
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=2,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Clean up SQL output
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)

# If the user inputs a question and there's a table schema, generate a query
if user_input and uploaded_file is not None:
    st.write("🤖: Let me analyze that...")
    generated_sql = generate_query(user_input, table_schema)
    st.write(f"Here's the SQL query based on your question: \n```{generated_sql}```")
else:
    if user_input and not uploaded_file:
        st.write("🤖: I can help with SQL queries, but first please upload a PDF with your table schema.")

# Friendly bot responses for common phrases
if "thanks" in user_input.lower():
    st.write("🤖: You're welcome! Let me know if you need more help.")
if "bye" in user_input.lower():
    st.write("🤖: Goodbye! Have a great day!")

