import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Function for API configuration
def sidebar_api_key_configuration():
    st.sidebar.subheader("API Keys")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key 🗝️", type="password", help='Get API Key here')
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key 🗝️", type="password", help='Get API Key here')

    if not all([openai_api_key, groq_api_key]):
        st.sidebar.warning('Enter both API Keys')
        st.session_state.prompt_activation = False
    elif valid_keys(openai_api_key, groq_api_key):
        st.sidebar.success('Keys valid. Ready to proceed!')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Invalid API keys')
        st.session_state.prompt_activation = False

    return openai_api_key, groq_api_key


def valid_keys(openai_key, groq_key):
    return openai_key.startswith('sk-') and groq_key.startswith('gsk_')

# Model Selection in sidebar
def sidebar_groq_model_selection():
    st.sidebar.subheader("Model Selection")
    return st.sidebar.selectbox('Select Model', ('Llama3-8b', 'Llama3-70b', 'Mixtral-8x7b', 'Gemma-7b'))
