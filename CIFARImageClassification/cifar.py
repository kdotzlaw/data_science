'''
Classify images using a CNN
CIFAR-10 dataset includes:
- training set of 50,000 images
- test set of 10,000 images
- classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship truck
'''
import tensorflow as tf
from keras import datasets, layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

'''
INPUT:
OUTPUT: returns the compiled model
PROCESS:
- convoluted layers extract spatial features
- add maxPooling to reduce feature map size
- use dropout to prevent overfitting by disabling neurons randomly when training
- add softmax layer that gets probability scores for classes
'''
def cnn():
    model = models.Sequential()
    # add layers
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # flatten layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation='softmax'))

    # compile the model with adam optimizer, categorical_crossentropy loss function, & accuracy metric
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model


if __name__=="__main__":
    # load cifar-10 datasets
    (trainX, trainY),(testX,testY) = datasets.cifar10.load_data()

    # normalize pixels to range [0,1] to improve model stability
    trainX = trainX.astype('float32')/255.0
    testX = testX.astype('float32')/255.0

    # encode labels into 10 dimensional vector to do multiclass classification
    trainY = to_categorical(trainY,10)
    testY = to_categorical(testY,10)

    # visuals of sample images in 4x4 grid, labeled with class name
    names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(trainX[i])
        plt.xlabel(names[np.argmax(trainY[i])])
    plt.show()

    # build the CNN model
    model = cnn()

    '''
    Train the model with:
    - epochs: 30 cycles of training
    - batch_size: 64 images per batch for best gradient updates
    - validation_split: helps with overfitting & generalization
    Store accuracy & loss results for plotting later
    '''
    history = model.fit(trainX,trainY,epochs=30, batch_size=64,validation_split=0.2)

    plt.figure(figsize=(12,5))
    # plot model accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'],label='Train Accuracy')
    plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plot model loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label='Train Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # predict class labels
    for i in range(5):
        # get img from testX
        img = testX[i]
        # get true class label from testY
        tLabel = names[np.argmax(testY[i])]

        # calc model prediction (the index of the label)
        pred = model.predict(np.expand_dims(img,axis=0))

        # get predicted label
        pLabel = names[np.argmax(pred)]

        # show results
        plt.imshow(img)
        plt.title(f"Actual: {tLabel} | Predicted: {pLabel}")
        plt.axis('off')
        plt.show()
    