import streamlit as st
import whisper
import tempfile
import os
import ffmpeg

# Load Whisper model
model = whisper.load_model("small")

def transcribe_audio(file_path):
    # Use whisper to transcribe audio
    result = model.transcribe(file_path)
    return result['text']

def convert_audio(input_file, output_file):
    # Use ffmpeg-python to convert the audio file if necessary
    try:
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file)
        ffmpeg.run(stream)
        return output_file
    except Exception as e:
        raise RuntimeError(f"Failed to convert file: {e}")

def save_uploaded_file(uploaded_file, file_type):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

# Streamlit app layout
st.title("Voice-to-Text Converter")
st.write("Upload your audio file (e.g., .m4a, .mp3, .wav, .ogg) and get the text transcription.")

# Supported file types
supported_types = ["m4a", "mp3", "wav", "ogg", "flac", "aac", "wma"]

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=supported_types)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Save the uploaded file
    file_path = save_uploaded_file(uploaded_file, file_extension)
    
    # Convert audio if necessary (e.g., if not supported directly by Whisper)
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    if file_extension != '.wav':
        file_path = convert_audio(file_path, output_file)

    # Display the audio player
    st.audio(file_path, format=f"audio/{file_extension.lstrip('.')}")
    
    # Transcribe button
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            try:
                # Perform transcription
                transcription = transcribe_audio(file_path)
                st.success("Transcription completed!")
                st.text_area("Transcription", transcription, height=200)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Cleanup temporary files
    os.remove(file_path)
    if os.path.exists(output_file):
        os.remove(output_file)
