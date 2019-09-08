# authors: @henardoel and @marloquemegusta
import tensorflow as tf
from keras.models import load_model
import argparse
import pickle
import cv2
from imutils.video import VideoStream
import time
import numpy as np

bottomlimit = np.array([120, 100, 100])
upperlimit = np.array([135, 255, 255])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
args = vars(ap.parse_args())

# initialize the array where the predictions are going to be kept,
# the size, and the index for frame counting
predictions = []
s = (96, 96, 3)
i = 0

# load the trained convolutional neural network and the label binarizer
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# main loop
while True:
    # initialize the mask with the size previously defined
    maskJpg = np.zeros(s)
    # extract the frame from the video stream
    frame = vs.read()
    # frame is converted to hsv for easier hue management
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv frame is blurred so high frequency noise is reduced
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)
    # pixel wise mask is created according to color thresholds
    mask = cv2.inRange(blurred, bottomlimit, upperlimit)
    # mask is processed to reduce noise and close little holes
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=4)
    # the mask is processed so it has the right shape to be introduced
    # to our trained network to be classified
    mask = cv2.resize(mask, (96, 96))
    mask = mask.astype("float") / 255.0
    maskJpg[:, :, 0] = mask
    maskJpg[:, :, 1] = mask
    maskJpg[:, :, 2] = mask
    maskJpg = tf.compat.v1.keras.preprocessing.image.img_to_array(maskJpg)
    maskJpg = np.expand_dims(maskJpg, axis=0)
    # classify the mask
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
            dist = nltk.FreqDist(lastPredictions)
            text = label(dist.max())
            cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        i += 1
    # the image is shown with its corresponding prediction
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
