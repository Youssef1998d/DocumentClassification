# Necessary packages
# import the necessary packages
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import tensorflow as tf
# to calculate accuracy
from sklearn.metrics import accuracy_score
BASE_DIR = Path(__file__).resolve().parent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def image_to_feature_vector(image, size=(128, 128)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size)



# construct the argument parse and parse the arguments

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("cnn_data/full_cnn_data/"))
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []

labels = []
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label (assuming that our
			# path as the format: /path/to/dataset/{class}.{image_num}.jpg
			
			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
			image = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			image = image.reshape( image.shape + (1,) ) 
			print(image.shape,"shape nparray255")
			label = imagePath.split("/")[-1].split("_")[0]
			
			
			# extract raw pixel intensity "features", followed by a color
			# histogram to characterize the color distribution of the pixels
			# in the image
			
			pixels = image_to_feature_vector(image)
			print(pixels.shape,"show image")
			pixels = pixels.reshape( pixels.shape + (1,) ) 
			print(pixels.shape,"shape nparray255")
			
			# update the raw images, features, and labels matricies,
			# respectively
			rawImages.append(pixels)
			
			labels.append(label)
			
			# show an update every 1,000 images
			if i > 0 and i % 1000 == 0:
				print("[INFO] processed {}/{}".format(i, len(imagePaths)))
# show some information on the memory consumed by the raw images
# matrix and features matrix
new_labels = []
class_label = {"bon" :0 , "devis" : 1 ,"facture" : 2 , "lettre" : 3, "cheque":4   }

for label in labels:
	new_labels.append(class_label[label]) 
	print(label)
print(labels[:10] , "oldlabels")
labels = new_labels
print(labels[:10] , "labels")
print(set(labels) , "unique labels")


# le = preprocessing.LabelEncoder()
# le.fit(labels)
# print(list(le.classes_) , "le classes")

# labels = le.transform(labels)
# print(labels[:20])
# {"lettre": 5 , "carte":0,"cv" : 3 , "credits" : 2 ,"cheques" : 1 , "facture" : 4 }
rawImages = np.array(rawImages)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))

print(len(rawImages), len(labels), "len raw w label")
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
trainRI = trainRI.astype('float32')
testRI = testRI.astype('float32')
trainRI /= 255
testRI /= 255
print(trainRI.shape,trainRL.shape , "shapes")

print("Shape before one-hot encoding: ", trainRL.shape)
print(testRI.shape,testRL.shape , "shapes")
trainRL = tf.keras.utils.to_categorical(
    trainRL, num_classes=5, dtype='float32'
)
testRL = tf.keras.utils.to_categorical(
    testRL, num_classes=5, dtype='float32'
)


print("Shape after one-hot encoding: ", trainRL.shape,trainRL[:12])
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(128, 128, 1)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(5, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(BASE_DIR,"checkpoints"),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
   )
model.fit(trainRI, trainRL, batch_size=64, epochs=8, validation_data=(testRI, testRL),callbacks=[model_checkpoint_callback],workers=6)
model.save('cnn_classification_model.h5')
# model = KNeighborsClassifier(n_neighbors=2,
# 	n_jobs=2)
# model.fit(trainRI, trainRL)
# acc = model.score(testRI, testRL)
# print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
# # train and evaluate a k-NN classifer on the histogram
# # representations
# # print("[INFO] evaluating histogram accuracy...")
# # model = KNeighborsClassifier(n_neighbors=3,
# # 	n_jobs=2)
# # # model.fit(trainFeat, trainLabels)
# # # acc = model.score(testFeat, testLabels)
# # # print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

# image = cv2.imread("chequetest.png")
# image = image_to_feature_vector(image)

# image = np.array(image)
# print(model.predict([image]) , "prediction")
