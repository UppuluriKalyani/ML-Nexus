# Sunglasses_Detection
Data Collection: The code scrapes face image data from a specific website (https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces/) and downloads the images to a local directory.
Data Preprocessing: It resizes the downloaded images to a uniform size of 32x30 pixels and converts them to grayscale. It also parses labels from filenames to determine if the person in the image is wearing sunglasses or not.
Neural Network Architecture: The code defines a simple neural network architecture with an input layer, a hidden layer, and an output layer. The network is trained to classify whether a person is wearing sunglasses based on the input face images.
Training the Model: The neural network is trained on the preprocessed face image data. It uses backpropagation to update weights and biases iteratively and minimize the error between predicted and actual labels.
Model Evaluation: After training, the code evaluates the accuracy of the trained model on a separate test set of images. It calculates the accuracy by comparing predicted labels with ground truth labels.
Model Saving and Loading: Once trained, the model is saved to disk using the pickle library. It can later be loaded for inference on new data.
Inference on Sample Data: The code demonstrates how to load the saved model and perform inference on a sample dataset to predict whether individuals in the images are wearing sunglasses.
Visualizing Predictions: It includes functions to display images along with their predicted and actual labels for visual inspection.
