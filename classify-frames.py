# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
import tensorflow as tf
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
from imutils import paths
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
hola1=0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--images", required=True,
	help="path to input images")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer
predictions=[]
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())
i=0
# load the images
imagePaths = list(paths.list_images(args["images"]))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
  
# pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = tf.compat.v1.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    predictions.append(proba)
    if proba[idx]>0.75:
        print(str(i)+"-"+label+"-"+imagePath.split(os.path.sep)[-1])
    i+=1
