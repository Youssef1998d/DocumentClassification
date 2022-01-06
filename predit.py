from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from pathlib import Path
from sklearn import preprocessing
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import tensorflow as tf
# to calculate accuracy
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent

def image_to_feature_vector(image, size=(64, 64)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size)

model = load_model("cnn_classification_model.h5")
model.summary()



image = cv2.imread("cheque771.png")


image =  cv2.resize(image, (128, 128))
print(image.shape,"shape resize128")

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
image = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
				

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
# image = cv2.threshold(
#                 image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# image = image.reshape( image.shape + (1,) ) 
# print(image.shape,"shape binairization")

image = np.array(image)
image = image.astype('float32')
image /= 255
print(image.shape,"shape nparray255")
image = image.reshape((1,) + image.shape) 
print(image.shape,"shape nparray255")

class_label = {"0" :"bon" , "1" : "devis" ,"2" : "facture" , "3" : "lettre", "4":"cheque"   }


print(class_label[str(np.argmax(model.predict(image)[0], axis=None , out = None) )], "prediction")
