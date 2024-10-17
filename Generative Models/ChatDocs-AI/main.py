import streamlit as st
from config import sidebar_api_key_configuration, sidebar_groq_model_selection
from pdf_processing import create_vectorstore
from chat_handler import get_llm_response
from streamlit_option_menu import option_menu
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Page Configuration
st.set_page_config(page_title="ChatDocs AI", page_icon=":robot_face:", layout="centered")

# Session State Variables
for key in ["vector_store", "response", "prompt_activation", "conversation", "chat_history", "prompt"]:
    if key not in st.session_state:
        st.session_state[key] = None

openai_api_key, groq_api_key = sidebar_api_key_configuration()
model = sidebar_groq_model_selection()

# Main App Interface
st.title("ChatDocs AI :robot_face:")
st.write("*Interrogate Documents, Ignite Insights*")

selected = option_menu(menu_title=None, options=["ChatDocs AI", "Reference", "About"], icons=["robot", "bi-file-text-fill", "app"], orientation="horizontal")
llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)
prompt_template = ChatPromptTemplate.from_template("Answer based on provided context only: {context} Questions: {input}")

# ChatDocs AI Section
if selected == "PDF ChatDocs AI":
    st.subheader("Upload PDF(s)")
    pdf_docs = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True, disabled=not st.session_state.prompt_activation)
    process = st.button("Process", type="primary", disabled=not pdf_docs)

    if process:
        st.session_state.vector_store = create_vectorstore(openai_api_key, pdf_docs)
        st.session_state.prompt = True
        st.success('Database is ready')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input(placeholder="Ask a document-related question", disabled=not st.session_state.prompt):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner('Processing...'):
            st.session_state.response = get_llm_response(llm, prompt_template, question)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response['answer']})
            st.chat_message("assistant").write(st.session_state.response['answer'])
