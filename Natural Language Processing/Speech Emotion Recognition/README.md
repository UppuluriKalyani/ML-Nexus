# Speech Emotion Recognition (SER) using Deep Learning

This project implements a **Speech Emotion Recognition (SER)** system using deep learning. It analyzes speech data and classifies it into different emotional states such as happy, sad, angry, neutral, etc. The model uses **CNNs (Convolutional Neural Networks)** to extract features from audio signals and predict emotions.

## Features
- **Data Augmentation**: Utilizes stretching, pitch shifting, and noise injection for robust training.
- **Preprocessing**: MFCC extraction for audio features.
- **CNN Model**: Built using Keras to classify speech emotions.
- **Training & Evaluation**: Model training on augmented data, with accuracy and loss metrics visualization.

## Datasets
The project uses four open-source datasets: **RAVDESS**, **CREMA-D**, **SAVEE**, and **TESS**, which are downloaded on the go.

## Model Architecture
- **Conv1D layers** to capture temporal features.
- **MaxPooling1D** for downsampling.
- **Dense layers** for final classification.
- **Dropout** to prevent overfitting.
