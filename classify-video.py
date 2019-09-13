# authors: @henardoel and @marloquemegusta
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

bottomLimit = np.array([120, 100, 100])
upperLimit = np.array([135, 255, 255])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-s", "--size", nargs=2, required=True)
args = vars(ap.parse_args())

# initialize the array where the predictions are going to be kept,
# the size, and the index for frame counting
predictions = []
i = 0
sp = SimplePreprocessor(96, 96)
# load the trained convolutional neural network and the label binarizer
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# main loop
while True:
    frame = vs.read()
    maskJpg= sp.maskExtraction(frame, bottomLimit, upperLimit)
    proba = model.predict(maskJpg)[0]
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
