# Eye Disease Classification Using CNN and MobileNetV2

This project implements a Convolutional Neural Network (CNN) and MobileNetV2 model for classifying different types and grades of eye diseases. The project focuses on using deep learning techniques, data augmentation, and evaluation metrics to classify images effectively.

## Project Overview

The classification of eye diseases is crucial for early diagnosis and treatment. This project aims to automate the detection and classification of eye diseases using deep learning models. We explore different architectures, focusing primarily on CNN and MobileNetV2, to identify multiple categories and severity levels of diseases.

## Features

- **Convolutional Neural Network (CNN)**: Custom CNN model built for feature extraction and classification of eye diseases.
- **MobileNetV2**: A pre-trained MobileNetV2 model is fine-tuned for high-accuracy classification.
- **Data Augmentation**: Techniques such as rotation, flipping, zooming, and shifting are applied to improve model generalization.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix are used to assess model performance.
- **Visualizations**: Visualize data and model performance through plots and images for better insights.

## Dataset

The dataset consists of images from multiple categories of eye diseases, including different grades and types. The dataset is split into training, validation, and test sets.

- **Categories**: List the categories or types of eye diseases in the dataset.
- **Grades**: Severity levels of the diseases.

## Project Structure

\`\`\`bash
├── data
│   ├── train
│   ├── validation
│   └── test
├── notebooks
│   ├── Data_Augmentation.ipynb
│   ├── CNN_Model_Building.ipynb
│   ├── MobileNetV2_Finetuning.ipynb
│   └── Model_Evaluation.ipynb
├── models
│   ├── cnn_model.h5
│   ├── mobilenetv2_model.h5
│   └── model_checkpoints
├── images
│   └── sample_images.png
├── README.md
└── requirements.txt
\`\`\`

## How to Run the Project

1. **Clone the repository**:

   \`\`\`bash
   git clone https://github.com/your-username/eye-disease-classification.git
   \`\`\`

2. **Install the required dependencies**:

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Download the dataset**:

   Download the dataset and organize it into the \`data\` folder following the structure below:

   \`\`\`bash
   ├── data
   │   ├── train
   │   ├── validation
   │   └── test
   \`\`\`

4. **Run the Jupyter notebooks** for data preprocessing, model building, and evaluation:

   - Data Augmentation: \`notebooks/Data_Augmentation.ipynb\`
   - CNN Model Building: \`notebooks/CNN_Model_Building.ipynb\`
   - MobileNetV2 Fine-Tuning: \`notebooks/MobileNetV2_Finetuning.ipynb\`
   - Model Evaluation: \`notebooks/Model_Evaluation.ipynb\`

5. **Training and Evaluation**:

   You can run the notebooks one by one to train the model, fine-tune MobileNetV2, and evaluate the model performance.

## Results

- **CNN Model**: Achieved an accuracy of 0.7491 on the training set and 0.7269 on the test set.
- **MobileNetV2**: Achieved an accuracy of 0.9170 on the training set and 0.8377 on the test set.

Visualizations of model performance, including confusion matrices and classification reports, are included in the evaluation notebook.

## Future Improvements

- Explore different architectures such as ResNet or EfficientNet for better accuracy.
- Increase the dataset size to improve generalization.
- Experiment with hyperparameter tuning, such as learning rates, optimizer choices, and batch sizes.

## Conclusion

This project demonstrates the potential of deep learning models in automating eye disease classification. By leveraging CNNs and MobileNetV2, we aim to improve the speed and accuracy of diagnosis, which can be crucial for medical professionals in treating patients.
