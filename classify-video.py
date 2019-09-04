# import the necessary packages
import tensorflow as tf
#from keras.preprocessing.frame import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
from imutils.video import VideoStream
import time
import numpy as np


bottomlimit=np.array([120,100,100])
upperlimit=np.array([135,255,255])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the label
# binarizer

predictions=[]
s=(96,96,3)

model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())
i=0
#initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
while True:
    maskJpg=np.zeros(s)
    frame=vs.read()
# pre-process the frame for classification
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)
    mask = cv2.inRange(blurred, bottomlimit, upperlimit)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=4)
    mask = cv2.resize(mask, (96, 96))
    mask = mask.astype("float") / 255.0
    maskJpg[:,:,0]=mask
    maskJpg[:,:,1]=mask
    maskJpg[:,:,2]=mask
    maskJpg = tf.compat.v1.keras.preprocessing.image.img_to_array(maskJpg)
    maskJpg = np.expand_dims(maskJpg, axis=0)
    # classify the input frame
    proba = model.predict(maskJpg)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    
    if proba[idx]>0.9:
        predictions.append(idx)
        if i>10:
            lastPredictions=predictions[-10:]
            dist=nltk.FreqDist(lastPredictions)
            text=label(dist.max())
            cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        i+=1
    #cv2.imshow("Frame", cv2.resize(mask,(512,512)))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
