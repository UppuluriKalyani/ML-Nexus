# Leaf Disease Detection using Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Data Preparation](#data-preparation)
   - [Model Building](#model-building)
   - [Training the Model](#training-the-model)
   - [Evaluation and Testing](#evaluation-and-testing)
   - [Saving the Model](#saving-the-model)
3. [Results and Analysis](#results-and-analysis)
4. [Discussion](#discussion)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction
In the realm of agriculture, plant diseases present a significant challenge, affecting crop yields and threatening food security worldwide. Early detection and treatment of plant diseases are crucial for sustainable farming. The traditional methods of detecting plant diseases involve manual observation by experts, which is time-consuming, subjective, and not scalable for large-scale farms. 

This project aims to address these challenges by leveraging deep learning and computer vision techniques to automate leaf disease detection. The objective is to build a convolutional neural network (CNN) model using transfer learning to classify various leaf diseases from a dataset of leaf images. The proposed solution utilizes a pre-trained VGG16 model, combined with custom layers for classification, to enhance performance in identifying and classifying different plant diseases. Data augmentation and transfer learning techniques are used to increase the accuracy and generalization of the model.

## Methodology

### Data Preparation
The dataset for this project consists of a collection of leaf images from the PlantVillage dataset, which contains images of healthy and diseased plant leaves across multiple species. To improve the model's generalization capabilities, data augmentation techniques were applied to the images, including:
- Rescaling: Normalizing pixel values to the range [0, 1].
- Rotation: Randomly rotating images within a range of 20 degrees.
- Width and Height Shifts: Randomly shifting images horizontally and vertically by up to 20%.
- Shear Transformations: Applying random shear transformations with a 0.2 shear intensity.
- Zoom Transformations: Randomly zooming in on images by up to 20%.
- Horizontal Flips: Randomly flipping images horizontally.
- Validation Split: Dividing the dataset into training (80%) and validation (20%) subsets.

The data was loaded using `ImageDataGenerator`, which facilitated these augmentations and ensured the model received diverse training samples.

### Model Building
To efficiently classify the leaf diseases, we used a transfer learning approach with the VGG16 model pre-trained on the ImageNet dataset. The VGG16 model serves as a feature extractor, with its weights frozen to prevent retraining. The following modifications were made to adapt the pre-trained model for the leaf disease detection task:
1. **Base Model**: The VGG16 model (excluding the top fully connected layers) was used as a base for transfer learning. Its weights were frozen to retain the feature extraction capabilities learned from the ImageNet dataset.
2. **Custom Layers**: Additional layers were added on top of the VGG16 base model:
   - A Flatten layer to convert the 3D output from VGG16 to a 1D vector.
   - A Dense layer with 256 neurons and ReLU activation function for learning complex patterns.
   - A BatchNormalization layer to normalize activations and improve training speed.
   - A Dropout layer with a 50% rate to prevent overfitting.
   - Another Dense layer with 128 neurons, followed by batch normalization and dropout.
   - An output Dense layer with a number of neurons equal to the number of classes in the dataset, using the softmax activation function for multi-class classification.

The final model architecture was compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

### Training the Model
The model was trained using the training dataset, with a validation split to monitor the model's performance. The training parameters included:
- **Epochs**: 50 (to allow for sufficient learning)
- **Batch Size**: 32 (for mini-batch gradient descent)
- **Learning Rate Reduction**: A callback `ReduceLROnPlateau` was used to reduce the learning rate by a factor of 0.2 if the validation loss did not improve for three consecutive epochs, with a minimum learning rate threshold of 1e-6.

### Evaluation and Testing
After training, the model was evaluated using the validation dataset to measure its accuracy and generalization. The test accuracy was calculated, and training history was plotted to visualize the accuracy and loss trends over epochs.

### Saving the Model
To facilitate future predictions and deployments, the trained model was saved as a file (`leaf_disease_model.h5`), which can be loaded for inference tasks.

## Results and Analysis
The model achieved high accuracy on the validation set, indicating its effectiveness in detecting and classifying leaf diseases. The training and validation accuracy graphs showed consistent improvement over epochs, while the loss curves demonstrated a decreasing trend, signifying effective learning and optimization.
- **Training Accuracy**: Increased steadily, reaching values close to 100% after 50 epochs.
- **Validation Accuracy**: Showed an upward trend, with occasional fluctuations due to overfitting, managed using dropout and data augmentation techniques.
- **Confusion Matrix Analysis**: A confusion matrix can be plotted to identify misclassifications and better understand which diseases the model finds difficult to distinguish.

## Discussion
The project's success demonstrates the potential of deep learning in automating plant disease detection. However, several considerations and future directions could further improve the model:
- **Incorporating More Data**: Increasing the dataset size, particularly with images from different environments, would enhance model generalization.
- **Fine-Tuning the VGG16 Layers**: Allowing some layers of VGG16 to be trainable could potentially boost accuracy.
- **Using Advanced Architectures**: Exploring deeper architectures like ResNet or EfficientNet could yield better performance.
- **Real-Time Deployment**: The model could be integrated into mobile applications or drones for real-time field monitoring.

## Conclusion
This project demonstrated a practical approach to leaf disease detection using deep learning, achieving high accuracy through a combination of transfer learning, data augmentation, and fine-tuning. The proposed model can serve as a valuable tool for precision agriculture, enabling early intervention and reducing crop losses.

## References
1. B.V. Nikith, N.K.S. Keerthan, M.S. Praneeth, Dr. T Amrita. "Leaf Disease Detection and Classification." 1877-0509
2. Tammina, Srikanth. (2019). Transfer learning using VGG-16 with Deep Convolutional Neural Network for Classifying Images. International Journal of Scientific and Research Publications (IJSRP). 9. p9420. 10.29322/IJSRP.9.10.2019.p9420.
3. Chittabarni Sarkar, Deepak Gupta, Umesh Gupta, Barenya Bikash Hazarika,”Leaf disease detection using machine learning and deep learning Review and challenges”, 1568-4946
4. Mohitsingh, [Kaggle Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

