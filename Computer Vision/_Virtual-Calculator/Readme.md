# AI Calculator Using Computer Vision and GenAI

## Description
This project creates a gesture-controlled virtual calculator, allowing users to perform basic arithmetic and trigonometric calculations using hand gestures instead of a traditional keyboard or mouse. Users can wave their hands in front of a camera to recognize specific gestures linked to mathematical operations. For trigonometric functions, users can draw shapes (like triangles, squares, circles) to perform operations such as sine, cosine, and tangent calculations based on the shapes they create. An AI component interprets the shapes and executes the corresponding trigonometric operations.

## Task Overview
- **Hand Gesture Recognition**: Implement hand gesture recognition to control a virtual calculator for basic operations (addition, subtraction, multiplication, division).
  
- **Trigonometric Operations**: Extend gesture recognition to support trigonometric operations by detecting shapes (e.g., triangle for sine/cosine, square for area calculations).
  
- **AI Module Integration**: Connect an AI module (Google Generative AI) to interpret drawn shapes and calculate trigonometric values or perform other geometry-related operations.
  
- **Virtual Calculator Interface**: Display calculations and results on a virtual calculator interface using Streamlit.

## Features
- Gesture-based control for intuitive interaction.
- Basic arithmetic operations supported.
- Shape recognition for advanced trigonometric calculations.
- AI-driven interpretation of geometric shapes using Google Generative AI.
- Interactive user interface powered by Streamlit.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- TensorFlow or any AI framework for shape recognition
- Streamlit
- Google Generative AI

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Pranshu-jais/ML-Nexus/edit/main/Computer%20Vision/%20Virtual-Calculator.git
   
   ```
   
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage
1. Position your hands in front of the camera.
2. Use gestures to perform arithmetic operations.
3. Draw shapes for trigonometric calculations.
4. View results displayed on the virtual calculator interface.


![2024-10-05 16-44-41-69](https://github.com/user-attachments/assets/bd0af8d4-dd66-435d-b7c7-789c5d3209ca)
