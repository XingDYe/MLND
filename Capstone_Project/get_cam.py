#! /usr/bin/python
 
import argparse
import cv2
import numpy as np 
from scipy import ndimage
from keras.models import load_model, Model
import matplotlib.pylab as plt 

HEIGHT = 128
WIDTH = 128
def pretained_path_to_tensor(img_path):
    img = cv2.imread(img_path)
    image = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
    x = np.expand_dims(image, axis=0)
    # convert the format of the image 
    x = np.array(x, dtype='float64')
    return x


def new_model_and_weights(model_path):
    # get the model
    model = load_model(model_path)
    # get AMP layers weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    new_model = Model(inputs=model.input, 
                  outputs=(model.layers[-4].output, model.layers[-1].output))
    
    return new_model, all_amp_layer_weights


def model_CAM(img_path, model, all_amp_layer_weights):
    image = pretained_path_to_tensor(img_path)
    last_conv_output, pred_vect = model.predict(image)
    # change dimensions of last conv output to 12*12*128 
    last_conv_output = np.squeeze(last_conv_output)
    pred = np.argmax(pred_vect)
    amp_layer_weights = all_amp_layer_weights[:, pred] 

    mat_for_mult = ndimage.zoom(last_conv_output, (128/12, 128/12, 1), order=1)
    x = mat_for_mult.reshape((128*128, 128))
    final_output = np.dot(x, amp_layer_weights).reshape(128,128)

    return final_output, pred


def plot_CAM(img_path, ax, model, all_amp_layer_weights):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128), interpolation=cv2.INTER_CUBIC)
    ax.imshow(img, alpha=0.5)

    # get class activation map
    CAM, pred = model_CAM(img_path, model, all_amp_layer_weights)
    ax.imshow(CAM, cmap='jet',alpha=0.5)
    if pred:
        title = 'This is a Dog'
    else:
        title = 'This is a Cat'
    ax.set_title(title)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    # model_path = 'gap_model.h5'
    model, all_amp_layer_weights = new_model_and_weights(args.model_path)
    fig, ax = plt.subplots()
    plot_CAM(args.image_path, ax, model, all_amp_layer_weights)
    plt.show()

