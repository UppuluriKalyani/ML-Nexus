# Real-Time Object Counter

A computer vision application that performs real-time object detection and counting using YOLOv8, capable of detecting and tracking multiple objects in real-time.

## 🎯 Project Overview

Real-Time Object Counter is a Streamlit-based web application that uses YOLOv8 for object detection. It provides an intuitive interface for real-time object detection and counting through webcam feeds or video uploads.

![app](https://github.com/user-attachments/assets/ca4f20ca-2503-464a-b61c-d9ded35bcb94)

## ⚡ Features
- Real-time object detection and counting
- Support for both webcam and video file inputs
- GPU acceleration
- Adjustable confidence threshold (0.10-1.00)
- Multi-class object selection
- Live performance metrics
- Interactive dashboard with:
  - Object count
  - Frame rate monitoring
  - Class tracking

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Custom CSS
- **Backend**: Python
- **ML Model**: YOLOv8
- **Project Files**:
  ```
  real-time-object-counter/
  ├── app.py               # Main application
  ├── main.css             # UI styling
  ├── yolov8n.pt           # YOLO model
  ├── run.bat              # Launch script
  ├── requirements.txt     # Dependencies
  └── README.md            # Documentation
  ```

## 📊 Performance Metrics (Based on Demo)
- **Frame Rate**: 18.65 FPS
- **Object Tracking**: 8 objects simultaneously
- **Active Classes**: 2 (person, car)
- **Detection Results**:
  | Class  | Count |
  |--------|--------|
  | Person | 7      |
  | Car    | 3      |

## 🚀 Installation Steps

1. Clone the repository
```bash
git clone https://github.com/UppuluriKalyani/ML-Nexus.git
```

2. Navigate to the Computer Vision directory
```bash
cd ML-Nexus/Computer Vision
```

3. Enter the Real Time Object Counter folder
```bash
cd "Real Time Object Counter"
```

4. Launch the app:
```bash
# Double-click run.bat
# OR
streamlit run app.py
```

## 💻 Usage Guide

1. Access the web interface at `http://localhost:8501`

2. Configure Detection Settings:
   - Adjust confidence threshold (default: 0.30)
   - Enable GPU acceleration
   - Select object classes (e.g., person, car)
   - Choose input source:
     - Live webcam feed
     - Upload video file (supported: MP4, AVI, MOV, MPEG4)

3. View Real-time Results:
   - Object detection visualization
   - Performance metrics
   - Object count table

## 🎮 Example Configurations

From the demo screenshot:
```python
# Configuration used
confidence_threshold = 0.30
enabled_classes = ['person', 'car']
gpu_enabled = True
input_type = 'video'  # City.mp4
```

## 🔄 Current Performance
- GPU-enabled processing
- Confidence scores ranging from 0.61 to 0.90
- Multiple object tracking:
  - People detection with scores: 0.90, 0.78, 0.63, etc.
  - Vehicle detection with scores: 0.85, 0.61, 0.82