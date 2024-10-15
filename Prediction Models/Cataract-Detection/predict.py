import cv2
import tensorflow as tf
import os
import numpy as np
from build_model import model_tools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model=model_tools()
model_folder='checkpoints'
image='49.jpg'
img=cv2.imread(image)
session=tf.Session()
img=cv2.resize(img,(100,100))
img=img.reshape(1,100,100,3)
labels = np.zeros((1, 2))

#Create a saver object to load the model
saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))

#restore the model from our checkpoints folder
saver.restore(session,os.path.join(model_folder,'.\\'))

#Create graph object for getting the same network architecture
graph = tf.get_default_graph()

#Get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("add_4:0")

#create placeholders to pass the image and get output labels
im_ph= graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

#Inorder to make the output to be either 0 or 1.
network=tf.nn.sigmoid(network)

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {im_ph: img, label_ph: labels}
result=session.run(network, feed_dict=feed_dict_testing)
print(result)



