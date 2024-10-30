# Text-to-Music Generator

This folder contains a Streamlit-based web application that generates music from text descriptions using Meta's Audiocraft library and the MusicGen model.

## Features

- **Text Input**: Enter a textual description of the type of music you want to generate.
- **Duration Control**: Select the duration of the generated music (up to 20 seconds).
- **Music Generation**: Generates music based on the provided description and duration.
- **Audio Playback**: Listen to the generated music directly in the browser.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/UppuluriKalyani/ML-Nexus.git
    cd Text-to-Music Generator
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv music-env
    source music-env/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Open your web browser** and go to `http://localhost:8501` to access the app.

3. **Enter a description** of the type of music you want to generate in the text area.

4. **Select the duration** of the music using the slider.

5. **Click "Generate Music"** to create and listen to your music.

<img width="656" alt="Screenshot 2024-05-21 at 6 12 32 PM" src="https://github.com/langchain-tech/Musicgen-Text-to-Music/assets/100914015/da41fbea-6565-4ac7-a78f-559666ff4b6f">

## Requirements

- `streamlit`
- `audiocraft`
- `torchaudio`
- `scipy`

## Project Structure

- `app.py`: The main Streamlit application script.
- `requirements.txt`: The dependencies required to run the app.

## License

This project is licensed under the MIT License.
