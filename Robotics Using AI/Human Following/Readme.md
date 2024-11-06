## Human Following Landrover
This project leverages **TensorFlow Lite** and the **MobileNet SSD v1 (COCO)** model to detect the presence of a person within the camera frame. The robot calculates the distance between itself and the person to generate forward motion commands, allowing it to autonomously follow the detected individual. This robot operates on **Raspberry Pi** and **Arduino**, using AI for efficient real-time processing.

## Features

- **Human Detection**: Uses the **MobileNet SSD v1 (COCO)** model for object detection, focusing on human presence within the camera frame.
- **Distance Calculation**: Based on the detected person's location in the frame, the robot calculates the distance and generates appropriate movement commands.
- **Autonomous Navigation**: Using the distance to the detected person, the robot moves forward or adjusts its position to maintain a safe following distance.
- **TensorFlow Lite Optimization**: The model is optimized using **TensorFlow Lite** for fast performance on Raspberry Pi.

## Getting Started

These instructions will guide you through setting up the Human Following Robot on **Raspberry Pi** and **Arduino**.

### Prerequisites

- **Hardware**:
  - Raspberry Pi 4 or later
  - Arduino Uno or similar microcontroller
  - Camera module for Raspberry Pi (for human detection and distance calculation)
  - Motor drivers and other hardware for robot mobility
  
- **Software**:
  - **TensorFlow Lite** for Raspberry Pi
  - **Python 3.7+** on Raspberry Pi
  - **Arduino IDE** for programming the microcontroller
  - MobileNet SSD v1 (COCO) model in TensorFlow Lite format


