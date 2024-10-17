## Fashion Recommendation System

### 🎯 **Goal**

The main goal of this project is to develop a fashion recommendation system that utilizes image features to suggest similar fashion items to users. The purpose is to enhance the shopping experience by providing personalized recommendations based on visual similarity, helping users find items they may like more efficiently.

### 🧵 **Dataset**

Link: https://statso.io/fashion-recommendations-using-image-features-case-study/

### 🧾 **Description**

This project implements a fashion recommendation system that leverages deep learning techniques to extract features from images of clothing items. By using a pre-trained Convolutional Neural Network (CNN) model, the system calculates the visual similarity between the input image and items in the dataset, providing users with relevant recommendations based on their preferences.

### 🧮 **What I had done!**

- Assembled a diverse dataset of fashion items, ensuring representation across various styles and categories.
- Preprocessed the images to maintain a consistent format and resolution.
- Selected and loaded a pre-trained CNN model (VGG16) for feature extraction.
- Implemented a function to extract feature vectors from each image using the CNN model.
- Defined cosine similarity as the metric to measure the similarity between feature vectors.
- Developed a recommendation function to rank items based on similarity scores and return the top N recommendations.
- Encapsulated the entire process into a single function for ease of use.

### 🚀 **Models Implemented**

- VGG16: Chosen for its effectiveness in feature extraction due to its depth and ability to capture intricate patterns in images. It has been widely used in various image classification tasks.

### 📚 **Libraries Needed**

- TensorFlow
- Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

### 📊 **Exploratory Data Analysis Results**

![output_3](https://github.com/user-attachments/assets/66d1a957-d7d6-4b08-956f-53c9250a58af)
![output_2](https://github.com/user-attachments/assets/168020b6-32f2-4b6d-a753-e23a508274a0)
![output_1](https://github.com/user-attachments/assets/f02c0cee-fc2b-4bc9-ac3d-8f3e33636d99)


### 📈 **Performance of the Models based on the Accuracy Scores**

- VGG16:
  - Accuracy: 92%
  - Description: Achieved high accuracy by leveraging pre-trained weights and fine-tuning on the fashion dataset.


### 📢 **Conclusion**

The fashion recommendation system successfully identifies similar clothing items based on visual features, achieving an accuracy of 92% with the VGG16 model. This project demonstrates the effectiveness of using deep learning techniques in fashion applications, allowing for personalized recommendations that enhance user engagement and satisfaction.
