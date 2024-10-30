# 3D Pose Estimation

This project implements real-time **3D human pose estimation** using **Mediapipe** and **OpenCV**. The goal is to estimate 3D coordinates of human body joints (keypoints) from a video or live webcam feed and send this data to external applications (e.g., Unity) via **UDP** for real-time animations or simulations.

## Features
- Real-time 3D pose estimation from live video or webcam input.
- Visualizes the detected pose landmarks on the video feed.
- Sends 3D pose data (in JSON format) over a network using a UDP socket for integration with other applications.
- Flexible and easy to integrate with game engines, AR/VR applications, or robotics projects.

## Requirements

Make sure to have the following installed:
- Python 3.7+
- OpenCV
- Mediapipe
- Numpy
- Socket Programming (standard Python library)
- JSON (standard Python library)

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy
