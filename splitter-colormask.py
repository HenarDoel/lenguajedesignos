# authors: @henardoel and @marloquemegusta
# usage: run  splitter-colormask.py -v targetvideo.mp4 -o targetdirectory -r rotation -s targetHeight targetWidth
import argparse
import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video")
ap.add_argument("-o", "--outputdir", required=True,
                help="path of the output directory")
ap.add_argument("-r", "--rotation", default="0", required=False,
                help="rotation in degrees")
ap.add_argument("-s", "--resize", default="96,96", required=False, nargs=2,
                help="size of frames in case you want to resize them")
args = vars(ap.parse_args())

# set color thresholds
bottomlimit = np.array([130, 90, 10])
upperlimit = np.array([160, 255, 255])

# initialize index for image counting
i = 0

# Opens the Video file
cap = cv2.VideoCapture(args["video"])

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame is converted to hsv for easier hue management
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv frame is blurred so high frequency noise is reduced
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)
    # pixel wise mask is created according to color thresholds
    mask = cv2.inRange(blurred, bottomlimit, upperlimit)
    # mask is processed to reduce noise and close little holse
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=4)
    # the mask is saved to disk
    cv2.imwrite(args["outputdir"] + "/" + str(i) + ".jpg",
                imutils.rotate(cv2.resize(mask, (args["resize"][0], args["resize"][1])), int(args["rotation"])))
    i += 1
cap.release()
cv2.destroyAllWindows()
