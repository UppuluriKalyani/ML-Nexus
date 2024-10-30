import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Classifier", layout="wide")

# Include Bootstrap CSS
bootstrap = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
"""
st.markdown(bootstrap, unsafe_allow_html=True)
st.markdown("""<h1>Pet Face's Image Classifier Web App</h1>""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Classify Image", "About Author"],
        icons=["house", "cloud-upload", "person"],
        menu_icon="cast",
        default_index=0,
    )
    
model_path = 'model.keras'

@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)

model = load_keras_model(model_path)
labels = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Persian', 'Ragdoll', 
    'Russian Blue', 'Siamese', 'Sphynx', 'american bulldog', 'american pit bull terrier', 'basset hound', 
    'beagle', 'boxer', 'chihuahua', 'english cocker spaniel', 'english setter', 'german shorthaired', 
    'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'miniature pinscher', 
    'newfoundland', 'pomeranian', 'pug', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu', 
    'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier'
]

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_class(model, img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

if selected == "Home":
    st.markdown("""
    Welcome to the Pet Image Classifier! 
    - This is a web application that leverages a ResNet Convolutional Neural Network (CNN) type model trained on the Oxford-IIIT Pet Images Dataset to accurately classify images of pets. 
    - The application is built using different frameworks and libraries such as Streamlit, TensorFlow.
    - Navigate using the menu to classify images or learn about the author.
    """)
elif selected == "Classify Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=300)
        st.markdown(
            """
            <style>
            .stImage {
            display: flex;
            justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("Classify Image"):
            predicted_class_label = predict_image_class(model, img)
            predicted_class_index = predicted_class_label[0]
            predicted_label = labels[predicted_class_index]
            st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="color: #4CAF50; font-size: 24px; font-weight: bold;">{predicted_label}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

elif selected == "About Author":
    st.markdown("""
    <h2>About the Author</h2>
    <p>This application was developed by <a href="https://github.com/yashksaini-coder">Yash K. Saini</a></p>
    <p>You can connect with him on the below social links</p>
    <div align='center'>
      <a href="mailto:ys3853428@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"></a>
      <a href="https://github.com/yashksaini-coder"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Github"></a>
      <a href="https://medium.com/@yashksaini"><img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium"></a>
      <a href="https://www.linkedin.com/in/yashksaini/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
      <a href="https://bento.me/yashksaini"><img src="https://img.shields.io/badge/Bento-768CFF.svg?style=for-the-badge&logo=Bento&logoColor=white" alt="Bento"></a>
      <a href="https://www.instagram.com/yashksaini.codes/"><img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white" alt="Instagram"></a>
      <a href="https://twitter.com/EasycodesDev"><img src="https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white" alt="X"></a>
    </div>
    """, unsafe_allow_html=True)
