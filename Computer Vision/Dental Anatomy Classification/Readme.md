# Dental Anatomy Classification and Detection with YOLOv9

## Overview

This project focuses on **dental number classification and detection** using the YOLOv9 model. It involves training a custom YOLO model on a dental dataset, testing the trained model on unseen data, and visualizing the predictions by drawing bounding boxes and class labels on the images.

The project is structured into two primary components:

1. **YOLOTrainer**: Handles the training process of the YOLOv9 model, allowing customization of task type, model size, image size, epochs, batch size, learning rate, optimizer, and weight decay.
   
2. **YOLOTester**: Loads a pre-trained YOLO model and uses it to make predictions on new input images.

3. **draw_predictions**: A utility function that visualizes the model’s predictions by drawing bounding boxes and class labels on the images and saving the output.

---

## Features

- **Customizable YOLO Training**: 
  The `YOLOTrainer` class allows flexible configuration of various training parameters, including:
  - Task (e.g., detection)
  - Model architecture (e.g., `yolov9s.pt`)
  - Image size for input
  - Number of epochs and batch size
  - Optimizer (e.g., Adam) and learning rate
  - Weight decay and model name

- **Prediction and Testing**: 
  The `YOLOTester` class provides functionality to load a trained YOLO model and make predictions on new data. The results, including bounding boxes and class labels, are returned for further processing or visualization.

- **Prediction Visualization**: 
  The `draw_predictions` function allows you to visualize the predictions made by the YOLO model by drawing bounding boxes around the detected objects and adding the class labels with confidence scores. The annotated images are saved for later review.

---

## Dataset

The dataset for this project includes images of dental structures, where each image has annotations for specific dental numbers (teeth) that need to be classified and detected. The dataset is structured in a format compatible with YOLO's requirements and specified via a YAML file (`/kaggle/input/dental-anatomy-dataset`).

---

## Installation

### Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- OpenCV
- [YOLOv9](https://github.com/ultralytics/yolov9)

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

---
