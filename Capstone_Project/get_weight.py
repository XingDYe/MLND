import h5py 
import os
import cv2
from keras.models import *
from keras.preprocessing.image import *

import numpy as np
import random
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from keras import layers


# Load npz file containing image arrays
x_npz = np.load("../input/x_images.npz")
x_train = x_npz['arr_0']
# Load binary encoded labels for cats or dog: 0=cat 1=dog
y_npz = np.load("../input/y_labels.npz")
y_train = y_npz['arr_0']

# First split the data in two sets, 80% for training, 20% for Validation)
X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(x_train,y_train, test_size=0.2, random_state=1, stratify=y_train)

print(np.array(X_TRAIN).shape)
print(np.array(X_VAL).shape)
print(K.image_data_format())


#####################################
###
### Model Architecture
###
######################################

nb_train_samples = len(X_TRAIN)
nb_validation_samples = len(X_VAL)

batch_size = 100
epochs = 50


def get_model(img_width=128, img_height=128):
    
    model = Sequential()

    model.add(layers.Conv2D(32, (1, 1), input_shape=(img_width, img_height, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.AveragePooling2D(12,12))

    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='sigmoid'))
    # model.add(layers.BatchNormalization())

    return model

def training():
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(np.array(X_TRAIN), Y_TRAIN, batch_size=batch_size)
    validation_generator = val_datagen.flow(np.array(X_VAL), Y_VAL, batch_size=batch_size)


    model = get_model(img_width=128, img_height=128)

    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['acc'])

    model.summary()

    print("******Training*******")

    history = model.fit_generator(train_generator, 
                steps_per_epoch=nb_train_samples // batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // batch_size)
    
    model.save('gap_model.h5')
    return model, history

def plot_history(history):
    import matplotlib.pylab as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'blue', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_1.png')
    plt.figure()
    plt.plot(epochs, loss, 'blue', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Losses')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_2.png')
    plt.show()
    
# if __name__=='__main__':
#     # save the trained model and figure 
#     model, history = training()
#     plot_history(history)
