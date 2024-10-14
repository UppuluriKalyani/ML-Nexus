import os
import cv2
from config import *
import random

#tools for image processing and data handing.
class utils:
    image_count = []
    count_buffer=[]
    class_buffer=all_classes[:]
    def __init__(self):
        self.image_count = []
        self.count_buffer = []
        for i in os.walk(data_path):
            if len(i[2]):
                self.image_count.append(len(i[2]))
        self.count_buffer=self.image_count[:]

    # processing images into arrays and dispatch as batches whenever called.
    def batch_dispatch(self,batch_size=batch_size):
        global batch_counter
        if sum(self.count_buffer):

            class_name = random.choice(self.class_buffer)
            choice_index = all_classes.index(class_name)
            choice_count = self.count_buffer[choice_index]
            if choice_count==0:
                class_name=all_classes[self.count_buffer.index(max(self.count_buffer))]
                choice_index = all_classes.index(class_name)
                choice_count = self.count_buffer[choice_index]

            slicer=batch_size if batch_size<choice_count else choice_count
            img_ind=self.image_count[choice_index]-choice_count
            indices=[img_ind,img_ind+slicer]
            images = self.generate_images(class_name,indices)
            labels = self.generate_labels(class_name,slicer)

            self.count_buffer[choice_index]=self.count_buffer[choice_index]-slicer
        else:
            images,labels=(None,)*2
        return images, labels

    #gives one hot for the respective labels
    def generate_labels(self,class_name,number_of_samples):
        one_hot_labels=[0]*number_of_classes
        one_hot_labels[all_classes.index(class_name)]=1
        one_hot_labels=[one_hot_labels]*number_of_samples
        #one_hot_labels=tf.one_hot(indices=[all_classes.index(class_name)]*number_of_samples,depth=number_of_classes)
        return one_hot_labels

    # image operations
    def generate_images(self,class_name,indices):
        batch_images=[]
        choice_folder=os.path.join(data_path,class_name)
        selected_images=os.listdir(choice_folder)[indices[0]:indices[1]]
        for image in selected_images:
            img=cv2.imread(os.path.join(choice_folder,image))
            batch_images.append(img)
        return batch_images
