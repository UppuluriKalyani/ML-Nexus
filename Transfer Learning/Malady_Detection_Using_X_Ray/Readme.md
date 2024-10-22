# Malady Detection Using X-Ray

This project focuses on detecting diseases from X-ray images using two popular deep learning models: **VGG16** and **ResNet-18**. The objective is to help identify potential medical conditions from X-rays through automated image classification.

## Models Used

### 1. **VGG16**:
   - VGG16 is a deep Convolutional Neural Network (CNN) architecture known for its simplicity and effectiveness in image classification tasks.
   - It consists of 16 layers with weights, primarily convolutional layers followed by fully connected layers.
   - **Accuracy**: ~85% on the test set.

### 2. **ResNet-18**:
   - ResNet (Residual Networks) introduces skip connections, allowing the model to train deeper networks by preventing vanishing gradient problems.
   - ResNet-18 is a variant with 18 layers, optimized for faster training and better performance on image recognition tasks.
   - **Accuracy**: ~88% on the test set.

## Dataset
- The X-ray images used for this project contain various categories of diseases and healthy scans. The dataset was preprocessed to remove noise and ensure uniformity.

## Key Steps
1. **Data Preprocessing**:
   - The images were resized and normalized to fit the input requirements of both models.
   - Data augmentation techniques like rotation and flipping were applied to increase the diversity of training samples.

2. **Training**:
   - Transfer learning was used by initializing the models with weights pre-trained on ImageNet.
   - The models were fine-tuned on the X-ray dataset to detect specific diseases.

3. **Evaluation**:
   - The models were evaluated using accuracy, precision, recall, and F1-score to gauge their performance in detecting maladies.

