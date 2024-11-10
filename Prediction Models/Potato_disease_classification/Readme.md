# Potato Disease Classification with CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) model to classify diseases in potato plants from leaf images. The model is trained using image data preprocessing techniques, including data augmentation and normalization, to enhance performance. This tool aids in identifying various potato diseases efficiently and accurately.

## Problem Statement

Identifying potato diseases accurately is critical in agriculture to prevent crop loss and ensure high yield. Traditional methods can be slow and error-prone, particularly when scaled. This project aims to develop a machine learning model to automatically detect and classify potato leaf diseases, thereby supporting better disease management and decision-making in agriculture.

## Project Structure

- `data/`: Contains the dataset for training and testing.
- `notebooks/`: Jupyter notebooks documenting model development and testing.
- `src/`: Source code for data preprocessing, model building, training, and evaluation.
- `README.md`: Project documentation.

## Model Architecture

The model uses a CNN with:
- Convolutional and pooling layers for feature extraction
- Dense layers for classification
- Softmax activation for multi-class output

Data augmentation techniques like random flips and rotations are applied to increase robustness.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib

