# Face_Orientations_Predictor
Downloading Face Images: The code starts by downloading face images from a given URL. It traverses through the directories and subdirectories, downloading .pgm files that contain images of faces with different orientations (left, straight, up, down).
Data Preprocessing: After downloading the images, the code preprocesses them by resizing each image to a fixed size (32x30) and normalizing pixel values to the range [0, 1].
Data Labeling: The code extracts labels from file names, which indicate the orientation of the face in each image.
Neural Network Training: It then trains a neural network model using the preprocessed images and their corresponding labels. The neural network is a multi-layer perceptron (MLP) with a customizable number of hidden layers and units.
Evaluation: The trained model is evaluated on a separate test set to measure its accuracy in predicting face orientations.
Prediction on New Images: Additionally, there's functionality to predict the orientation of new face images. The code can download new face images from a specified URL and predict their orientations using the trained model.
Visualization: Finally, the code includes functions to visualize the predictions, displaying the images along with their predicted orientations.
