import kagglehub
from imutils import paths
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from keras.applications import VGG16
from keras.models import Model
from keras.optimizers import Adam




'''
INPUT:
OUTPUT:
PROCESS:
'''
def plotImg(image):
    plt.imshow(image)


'''
INPUT: training and testing data
OUTPUT:
DESC: 
- A convoluted NN is deep and used to analyze images
- Consists of convolution layer for feature extraction & fully connected layer
- Using VGG16 network model but removing last layer and adding layers for bt analysis
that uses ouput of convoluted layer to predict image class/label
PROCESS:
- 
'''
def cnn(trainX, trainY, testX, testY):
    # fill out boundary pixels with nearest pixels & add rotation of 15deg
    trainGenerator = ImageDataGenerator(fill_mode='nearest',rotation_range=15)

    # removing base layer & adding custom layers
    baseModel = VGG16(weights='imagenet',input_tensor=Input(shape=(224,224,3)), include_top=False)
    baseInput = baseModel.input
    baseOutput = baseModel.output
    baseOutput = AveragePooling2D(pool_size=(4,4))(baseOutput)
    baseOutput=Flatten(name='flatten')(baseOutput)
    baseOutput=Dense(64,activation='relu')(baseOutput)
    baseOutput=Dropout(0.5)(baseOutput)
    baseOutput = Dense(2,activation='softmax')(baseOutput)
    
    # freeze model layers so network doesnt start off trained (ie uses weights of prev layers & trains for the newly added layers)
    for layer in baseModel.layers:
        layer.trainable = False

    # build model & compile with Tensorflow's Adam (with learning rate=0.001, and metric=accuracy)
     # Since binary classification of images, binary cross entropy is loss function
    model = Model(inputs = baseInput, outputs=baseOutput)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'],
                  loss='binary_crossentropy')
   
    print(model.summary())





if __name__=='__main__':
    path = "C:\\Users\\Katrina\\.cache\\kagglehub\\datasets\\navoneel\\brain-mri-images-for-brain-tumor-detection\\versions\\1"
    # load image directory
    img_paths = list(paths.list_images(path))

    # iterate over each path and extract directory name (no/yes): this is label
    images = []
    labels = []
    for img_path in img_paths:
        label = img_path.split(os.path.sep)[-2]
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))

        images.append(img)
        labels.append(label)
    # plot an image to test
    #plotImg(images[0])

    # convert images[] and labels[] to np.arrays for normalizations
    iamges = np.array(images)/255.0
    labels = np.array(labels)

    # apply one-hot encoding to labels (ie transform to binary)
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    labels = to_categorical(labels)

    # split data into train and test (90/10)
    trainX, testX, trainY, testY = train_test_split(
                images, labels,
                 test_size=0.10,random_state=42,stratify=labels)
    
    # call cnn
    cnn(trainX, trainY, testX, testY)