import streamlit as st
import os
from speech_recognition import main
from text_sentimet import  text_analysis
# Main layout

# Set the page configuration
st.set_page_config(page_title="Priyanshu-Choubey", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)



# Page header
st.title("ViveVocal : Analyzing emotional Tone in Real-Time Speech")


st.header("""Choose how you'd like to proceed furthur:""", divider='rainbow')

# Option to select input method
input_method = st.radio("How would you like to provide input?", 
                        ('None','Voice Input', 'Select Pre-stored File','By entering text'),index=0)

# If user selects 'Voice Input'
if input_method == 'Voice Input':
    main()
    
    

# If user selects 'Select Pre-stored File'
elif input_method == 'Select Pre-stored File':
    st.info("You chose to use a pre-stored text file.")

    #provide your directory which contain a reviews folder
    file_directory = 'speech_text/'
    if not os.path.exists(file_directory):
        st.warning(f"The directory `{file_directory}` does not exist. Please ensure the folder is created.")
    else:
        files = [f for f in os.listdir(file_directory) if f.endswith('.txt')]
        if len(files) > 0:
            selected_file = st.selectbox("Choose a file", files)

            # Display the content of the selected file
            if selected_file:
                file_path = os.path.join(file_directory, selected_file)
                with open(file_path, 'r') as file:
                    content = file.read()
                    st.text_area("File Content", content, height=200)
                    if st.button(" Summarize Review"):
                        text_analysis(1,content)    
        else:
            st.warning(f"No .txt files found in the `{file_directory}` directory.")

elif input_method=="By entering text":
    st.info("You chose to input text manually.")

    # Text input box
    user_input = st.text_input("Enter your text here:")

    if user_input:
        # Display the manually entered text in a beautiful box
        st.markdown(f"""
        <div style="
            background-color: #f0f0f5;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            font-size: 16px;
            font-family: Arial, sans-serif;
            color: #333;
            text-align: center;
        ">
            {user_input}
        </div>
        """, unsafe_allow_html=True)
    text_analysis(2,user_input)