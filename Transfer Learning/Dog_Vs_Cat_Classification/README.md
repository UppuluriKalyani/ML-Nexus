# **Dogs vs. Cats Classification using Transfer Learning**  
![Status](https://img.shields.io/badge/Status-Completed-green)  
![Python](https://img.shields.io/badge/Python-3.x-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  

A Deep Learning project that classifies images of **dogs** and **cats** using **Transfer Learning** with **MobileNetV2** from TensorFlow Hub. This project demonstrates how pre-trained models can be leveraged to achieve high performance with minimal training time.  

---

## **Table of Contents**  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Common Errors](#common-errors)  
- [Results](#results)  
- [Contributing](#contributing)  
- [References](#references)  

---

## **Features**  
- Uses **MobileNetV2** from TensorFlow Hub for feature extraction.  
- **Binary classification**: Predict whether the input image is a **Dog** or a **Cat**.  
- Transfer Learning helps reduce training time while achieving excellent accuracy.  
- Easily deployable for real-world use in image classification tasks.  

## **Dataset**  
- **Dataset**: [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/overview)  
- **How to Use Kaggle API**:  
   1. **Create a Kaggle Account**: Go to [Kaggle](https://www.kaggle.com/) and sign up.  
   2. **Verify Your Account**: Verify your email or phone number to enable the API.  
   3. **Generate API Token**: Go to *Account Settings*, find the "Create New API Token" option, and click it. A `kaggle.json` file will be downloaded.  
   4. **Upload `kaggle.json` to Google Colab**: Use the following command to upload the file:  
      ```python
      from google.colab import files
      files.upload()  # Upload your kaggle.json here
      ```  
   5. **Access the Dataset Using API**: Once uploaded, you can download datasets using the Kaggle API.

> **Note**: The Dogs vs. Cats dataset is quite large. Instead of downloading it directly to your local machine, it's recommended to use the Kaggle API to access the dataset directly in your Google Colab environment.

## **Installation**  
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/dogs-vs-cats.git
   cd dogs-vs-cats
   ```

2. **Install Dependencies**  
   Ensure you have **Python 3.x** installed. Run:  
   ```bash
   pip install tensorflow tensorflow-hub matplotlib numpy scikit-learn Pillow
   ```

## **Common Errors**  

### Issue with `tf.keras.Sequential`  
While building the neural network, I encountered an issue using `tf.keras.Sequential`. If you face the same error, try the following:  

1. **Install tf_keras**  
   ```bash
   pip install tf_keras
   ```

2. **Modify Your Code**  
   Instead of:  
   ```python
   model = tf.keras.Sequential([...])
   ```  
   Use:  
   ```python
   import tf_keras  
   model = tf_keras.Sequential([...])
   ```  

This workaround ensures compatibility if TensorFlow’s default Keras module doesn’t work in your environment.  

## **Results**  
- **Test Loss**: 0.0445  
- **Test Accuracy**: 99.00%  

These results demonstrate that the model is highly accurate in distinguishing between images of dogs and cats.

## **Contributing**  
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open a pull request or an issue.  

## **References**  
- [TensorFlow Hub Documentation](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)  
- [TensorFlow Hub](https://www.tensorflow.org/hub)  
- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)  

## **Conclusion**  
The Dogs vs. Cats classification project successfully demonstrates the effectiveness of transfer learning for image classification tasks. The high accuracy indicates that the model generalizes well to unseen data. This approach can be extended to other classification problems with similar image datasets.
