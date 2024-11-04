# Chess Position FEN Prediction using CNN

This repository contains a Convolutional Neural Network (CNN) model designed for the prediction of chess board positions in FEN (Forsyth-Edwards Notation) format from images. The project explores exploratory data analysis (EDA) on chess board images and implements a CNN model to predict the respective FEN strings.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Overview

Chess position recognition is essential for building applications in chess analysis, broadcasting, and AI-based evaluation of games. This project leverages a CNN-based deep learning model to recognize and predict FEN notation, which captures the state of a chess game.

## Data

The dataset consists of images representing various chess board states. Each image is labeled with its corresponding FEN notation, which is parsed to train the model. Image preprocessing includes resizing, grayscale conversion, and normalization.

## Model Architecture

The model architecture is a deep Convolutional Neural Network designed to handle image input and output FEN strings, capturing board layout in terms of piece position and type.

### Layers

1. **Input Layer**: Accepts 2D image data (chess board).
2. **Convolutional Layers**: Multiple layers with increasing depth to capture board layout and distinguish piece types.
3. **Pooling Layers**: To down-sample the image spatially.
4. **Fully Connected Layers**: Converts 2D convolutional outputs into classification probabilities.
5. **Output Layer**: Maps to FEN string components.

### Loss and Optimization

- **Loss Function**: Categorical Cross-Entropy, suitable for multi-class classification.
- **Optimizer**: Adam Optimizer is used for efficient gradient-based learning.

## Preprocessing

Preprocessing steps:
1. **Image Resizing**: Images are resized to a standard dimension.
2. **Normalization**: Pixel values are normalized for consistency.
3. **Encoding**: FEN strings are encoded as multi-class labels for each board square.

## Training

Training involves:
- Splitting data into training, validation, and testing sets.
- Training the model over multiple epochs with mini-batches.
- Monitoring accuracy and loss.

## Evaluation

Model evaluation includes metrics such as:
- **Accuracy**: Fraction of correctly predicted board configurations.
- **Precision and Recall**: For capturing model effectiveness in piece placement accuracy.
- **Confusion Matrix**: Visual representation of prediction errors for different chess pieces.

