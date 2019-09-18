# authors: @henardoel and @marloquemegusta
# usage: run  classify-video.py -m trained_model -l label_binarizer -s targetHeight targetWidth
import tensorflow as tf
from keras.models import load_model
import argparse
import pickle
import cv2
from imutils.video import VideoStream
import time
import numpy as np
from scipy.stats import mode
from SimplePreprocessor import SimplePreprocessor

# set color thresholds
bottomLimit = np.array([120, 100, 100])
upperLimit = np.array([135, 255, 255])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-s", "--size", nargs=2, default=(96, 96), required=False)
args = vars(ap.parse_args())

# initialize the array where the predictions are going to be kept,
# the size, the index for frame counting and and object Simple Preprocessor
# from which we are going to import some functions
predictions = []
i = 0
sp = SimplePreprocessor(int(args["size"][0]), int(args["size"][1]))

# load the trained convolutional neural network and the label binarizer from the arguments
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# main loop
while True:
    frame = vs.read()
    # we call the Simple Preprocessor function "maskExtraction"
    maskRGB = sp.maskExtraction(frame, bottomLimit, upperLimit)

    # the image is converted to an array that tensorflow can understand
    maskRGB = tf.compat.v1.keras.preprocessing.image.img_to_array(maskRGB)

    # np.expand_dims is used to go from (height,width,3) to
    # (1, height, width, 3), that is the input format of our net
    maskRGB = np.expand_dims(maskRGB, axis=0)

    proba = model.predict(maskRGB)[0]
    idx = np.argmax(proba)

    # obtain the label associated with the prediction
    label = lb.classes_[idx]

    # the prediction's label is written in the shown image only if its probability
    # is higher than a certain value
    if proba[idx] > 0.9:
        predictions.append(idx)
        if i > 10:
            # predictions are being kept until there are 10 of them
            # then the most frequent one is the one that will be written in the image shown
            lastPredictions = predictions[-10:]
            text = lb.classes_[mode(lastPredictions)[0][0]]
            cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        i += 1
    # the image is shown with its corresponding prediction
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
