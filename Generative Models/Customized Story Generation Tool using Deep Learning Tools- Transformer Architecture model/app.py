import streamlit as st
from transformers import pipeline


generator = pipeline("text-generation", model="gpt2")


st.title("AI Story Generator")


prompt = st.text_input("Enter a prompt:")


max_length = st.slider("Select the maximum length of the story:", 
                       min_value=50, max_value=500, value=100, step=10)


if st.button("Generate Story"):
    if prompt:
        
        result = generator(prompt, max_length=max_length, num_return_sequences=1)
        
        st.write(result[0]['generated_text'])
    else:
        st.write("Please enter a prompt to generate a story.")
