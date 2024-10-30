# Age and Gender Prediction Using Functional API and Transfer Learning (VGG16)

This project aims to predict the age and gender of individuals using a deep learning model built with the Functional API and Transfer Learning techniques. The model leverages the VGG16 architecture pre-trained on ImageNet and is fine-tuned to perform age and gender classification.

## Dataset

The dataset used in this project is the UTKFace dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/jangedoo/utkface-new). The dataset contains over 23,000 images of faces with labels for age, gender, and ethnicity. The dataset has been preprocessed and split into training, and test sets.

## Key Features

- **Functional API with Multi-Output Model**: Utilizes TensorFlow's Functional API to create a complex model that can predict both age and gender simultaneously. The architecture allows for shared layers and multiple outputs, making the model efficient and flexible for handling multi-task learning.
- **Transfer Learning**: Leverages the pre-trained VGG16 model on ImageNet to boost performance, especially when working with limited data.
- **Data Augmentation**: Includes various techniques like rotation, flipping, and zooming to increase the diversity of the training data.
- **Early Stopping**: Monitors validation loss to prevent overfitting by stopping training when performance on the validation set starts to degrade.
- **Model Checkpointing**: Saves the best-performing model based on validation accuracy, ensuring that the best version of the model is retained for future use.

## Model Architecture

1. **VGG16 Base**:
   The pre-trained VGG16 model is loaded without the top layers, serving as the feature extractor.

3. **Custom Layers for Feature Extraction**:
   - After flattening the VGG16 output, dense layers are added for age and gender prediction.
   - Batch Normalization and Dropout Layers are used to stabilize training and prevent overfitting.
   - Callbacks like early stopping and model checkpointing help save the best model and avoid overfitting.

4. **Branching for Multi-Output**:
   - The architecture branches into two separate heads after the shared dense layers:
     - **Age Prediction Head**: This branch consists of additional dense layers followed by an output layer with a single neuron, which predicts the age as a continuous value.
     - **Gender Prediction Head**: This branch includes dense layers leading to a final output layer with a single neuron and a sigmoid activation function, predicting the gender as a binary value (0 for male, 1 for female).

5. **Loss Functions**:
   - Separate loss functions are used for the two outputs:
     - **Mean Absolute Error (MAE)** for age prediction, as age is a continuous variable.
     - **Binary Crossentropy** for gender prediction, as gender is a binary classification task.
   - The total loss is a weighted sum of these individual losses, allowing the model to learn both tasks simultaneously.

6. **Optimization and Metrics**:
   - The model is optimized using the Adam optimizer, known for its efficiency in handling sparse gradients on noisy problems.
   - Metrics such as Mean Absolute Error (MAE) for age and Accuracy for gender are tracked during training to monitor the model's performance on both tasks.

## Usage

### Predicting Age and Gender from an Image

You can use the trained model to predict the age and gender of a new image by loading the model and processing the image as described in the project code.

## Results

During training, the model achieved the following performance:

- **Age MAE**: 7.17 on the test set
- **Gender Accuracy**: 90.42% on the test set

## Conclusion
This project illustrates how the Functional API can be used to create sophisticated deep learning models capable of handling multiple outputs. By combining the flexibility of the Functional API with the strength of transfer learning, the model achieves robust performance in predicting both age and gender."

![Screenshot (58)](https://github.com/user-attachments/assets/21593b12-e8f4-4678-88b4-e3daabea51b3)
