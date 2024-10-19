import streamlit as st
from scrape import scrape_webiste , clean_body , get_html_body , make_batches
from parse import parse_with_llm
from bs4 import BeautifulSoup

st.title("WebWeaver : AI Web Scrapper")
url = st.text_input("Enter your website name")

if st.button("Scrape Site"):
    if url:
        st.write(f"Scraping the Website")
        results = scrape_webiste(url)
        soup = BeautifulSoup(results)
        # st.write(soup.prettify())
        body_text  = get_html_body(results)
        clean_text  = clean_body(body_text)
        
        st.session_state.dom_content = clean_text
        
        with st.expander("View Website Content"):
            st.text_area("Website Content" , clean_text , height= 350)
        

if "dom_content" in st.session_state:
    parse_descption = st.text_area("Describe what do you want to parse?")
    open_ai_api = st.text_input("Enter you openai API Key")
    
    if st.button("Parse Content"):
        st.write("Parsing the content")
        text_chunks = make_batches(st.session_state.dom_content)
        
        results = parse_with_llm(text_chunks , parse_descption , open_ai_api)
        st.write(results)
        
       
        
       
    