# Gesture-Based Keyboard Interface Using Webcam

## Project Overview
This project implements a virtual keyboard that can be controlled using hand gestures captured through a webcam. Users can type by making pinching gestures in the air, making typing possible without physical contact with any surface.

## Features
- ✋ Hand gesture-controlled virtual keyboard
- 👆 Pinch-to-type functionality
- ✊ Fist gesture to close keyboard
- ⌫ Backspace functionality
- 🎯 Reduced click sensitivity for accurate typing
- 💡 Visual feedback with lighting effects
- 🎨 Enhanced UI with centered keyboard layout
- 📝 Real-time text display

## Requirements
- Python 3.7+
- OpenCV
- CVZone
- Mediapipe
- NumPy
- PyAutoGUI

For exact versions, please refer to the `requirements.txt` file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaival111/Gesture-Based-Virtual-Keyboard.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python main.py
```

2. Gesture Controls:
   - Move your hand to hover over virtual keys
   - Pinch index finger and thumb to "press" keys
   - Make a fist to close the keyboard
   - Use the "BACK" button for backspace
   - Use the "CLOSE" button or make a fist to exit

## Technical Details

### Hand Detection
- Uses CVZone's HandTrackingModule
- Detection confidence threshold: 0.8
- Supports tracking of up to 2 hands

### Keyboard Layout
- QWERTY layout with special characters
- Additional control buttons (Backspace, Space, Close)
- Enhanced visual feedback system
