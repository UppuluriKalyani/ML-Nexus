import streamlit as st 
import tensorflow as tf 
import numpy as np 
from typing import List
import pathlib
curr_dir_path = pathlib.Path.cwd()
curr_dir_parent = curr_dir_path.parent
page_bg = """
<style>
    .stApp {
        background-image: url('https://images.pexels.com/photos/129731/pexels-photo-129731.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    textarea {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
</style>
"""

# Render the background CSS
st.markdown(page_bg, unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model
my_model = load_model(str(curr_dir_parent)+"/SkimLit_Models/Model-1.keras")

classes = ["BACKGROUND","OBJECTIVE","METHODS","RESULTS","CONCLUSIONS"]

def data_preprocessing(data: List[str]):
    dataset = tf.data.Dataset.from_tensor_slices((data))
    dataset=dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

def process_text(input_text):
    abstract_line_split = input_text.split('.')
    print(abstract_line_split[:-1],"HERE")
    return abstract_line_split[:-1]

st.title("SkimLit Text Processor")


input_text = st.text_area("Input Paragraph:", height=150)

# Create a button to process the text
if st.button("Process Text"):
    if input_text:
        text_list = process_text(input_text)
        data = data_preprocessing(text_list)
        curr_list=[[""] for _ in range(len(classes))]
        final_text=""
        start_point=0
        for batch in data:
            print(f"Length of batch {len(batch)}")
            prediction = my_model(batch)
            predicted_classes= tf.argmax(prediction,axis=1).numpy()
            for i in range(len(predicted_classes)):
                curr_list[predicted_classes[i]].append(text_list[start_point+i])
            start_point+=len(predicted_classes)
        background = classes[0]+(".").join(curr_list[0])
        objective = classes[1]+(".").join(curr_list[1])
        methods = classes[2]+(".").join(curr_list[2])
        results = classes[3]+(".").join(curr_list[3])
        conclusion = classes[4]+(".").join(curr_list[4])
        final_text=background+'\n'+objective+'\n'+methods+'\n'+results+'\n'+conclusion

        st.text_area("Output Paragraph:", value=final_text, height=150)
    else:
        st.error("Please enter a paragraph to process.")
