#!/usr/bin/python

import cv2
import os
import random
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# ../input/train/
SOURCE_IMAGES = os.path.join('input', "train")

# ../input/train/*.jpg
IMAGES = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

WIDTH = 128
HEIGHT = 128
NUM_CLASSES = 2

def get_files(IMAGES):  
    ''''' 
    Args: 
        IMAGES: list of all images 
    Returns: 
        list of images and matched labels 
    '''  

    Cats = []
    labels_cat = []
    Dogs = []
    labels_dog = []

    for img_path in IMAGES:
        base = os.path.basename(img_path) # get the name of each image
        name = base[0:3]
        if (name=='dog'):
            Dogs.append(img_path)

        else:
            Cats.append(img_path)
          
    labels_dog = np.ones((len(Dogs),))
    labels_cat = np.zeros((len(Cats),))

    print('There are %d Cats\nThere are %d Dogs' %(len(Cats), len(Dogs)))  
      
    image_list = np.hstack((Cats, Dogs))  
    label_list = np.hstack((labels_cat, labels_dog))  
    temp = np.array([image_list, label_list])  
    temp = temp.transpose()  
    np.random.shuffle(temp) 
    label_list = list(temp[:, 0])
    image_list = list(temp[:, 1])

    label_list = [int(i) for i in label_list]    
    return image_list, label_list  


def preprocess_image(image_list, HEIGHT, WIDTH):

	samples = len(image_list)
	for n in range(samples):
		image = image_list[n]
		full_size_image = cv2.imread(image)
		img = cv2.resize(full_size_image, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
		image_list[n] = img 
	return image_list

if __name__=='__main__':
	
	image_list, label_list = get_files(SOURCE_IMAGES)
	y_labels = to_categorical(label_list, num_classes=NUM_CLASSES)
	x_train = preprocess_image(image_list, HEIGHT, WIDTH)
	np.savez('x_images', x_train)
	np.savez('y_labels')

