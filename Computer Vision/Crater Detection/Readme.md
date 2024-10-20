# Crater Detection Model

This repository contains a machine learning model for detecting lunar craters, particularly focusing on low-light regions like the Permanently Shadowed Regions (PSR) of the moon. The model leverages advanced object detection techniques to improve surface mapping and assist in identifying potential landing sites on the lunar surface.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Accurate detection of lunar craters is essential for tasks such as identifying safe landing sites and analyzing the lunar surface. This crater detection model is designed to work efficiently even in low-light conditions found in PSR areas. By incorporating this model, lunar mapping missions can achieve higher precision and reliability.

## Dataset
The model was trained on the CIFAR-10 dataset and fine-tuned using lunar surface images to detect craters more effectively. The dataset consists of:

- Lunar surface images
- Ground truth annotations for craters

## Model Architecture
The crater detection model is built using a Convolutional Neural Network (CNN) for object detection and image segmentation. Key features include:

- **Input size**: 2048x1024 pixels (adjustable)
- **Object detection**: Identifies craters and other relevant features on the lunar surface
- **Low-light optimization**: Specifically fine-tuned for PSR areas
- **Evaluation metrics**: Precision, Recall, F1-score

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/<your_username>/ML-Nexus.git
    cd ML-Nexus/Computer Vision/Crater Detection
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install Git LFS (if using large files):**
    ```bash
    git lfs install
    ```

## Usage

1. **Run the Jupyter Notebook:**
    - Open the `crater_detection_model.ipynb` notebook and run all the cells to train and test the model.
    
    Alternatively, to run it from the terminal:

    ```bash
    jupyter notebook crater_detection_model.ipynb
    ```

2. **Training and Testing:**
    - Modify the dataset path or model parameters within the notebook if necessary.
    - Visualize the detected craters using the output provided at the end of the notebook.

## Results
The model effectively detects craters in high-resolution lunar images, especially in challenging low-light areas. Below is a sample visualization:

| Input Image | Crater Detection Output |
| ----------- | ----------------------- |
| ![input](images/input_sample.jpg) | ![output](images/output_sample.jpg) |

## Contributing
We welcome contributions to enhance the model or add new features. Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
