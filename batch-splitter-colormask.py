# authors: @henardoel and @marloquemegusta
# usage: this script grabs any videos from the subfolders on the -v directory.
# then, it splits this videos into frames and save them on the subfolders of
# -o directory where the name of the subfolder matches with the name of the
# subfolder on -v. -r corresponds to rotation, -d to decimate and -s to resize.

import argparse
import cv2
import imutils
import os
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videos", required=True,
                help="path to input videopath")
ap.add_argument("-o", "--outputdir", required=True,
                help="path of the output directory")
ap.add_argument("-r", "--rotation", default="0", required=False,
                help="rotation in degrees")
ap.add_argument("-d", "--decimate", default="1", required=False,
                help="saves only 1 out of each -d frames")
ap.add_argument("-s", "--resize", default="96,96", required=False, nargs=2,
                help="size of frames in case you want to resize them")
args = vars(ap.parse_args())

# set color thresholds
bottomlimit = np.array([120, 100, 10])
upperlimit = np.array([150, 255, 255])

# initialize the array where the paths to the videos are going to be kept
videoPaths = []

# an array that contains the paths of all files on -v and subfolders is created
# r=root, d=directories, f = files
for r, d, f in os.walk(args["videos"]):
    for file in f:
        videoPaths.append(os.path.join(r, file))

# initialize index for image and frame counting
i = 0
framecount = 0

# Opens the Video file
for videoPath in videoPaths:
    cap = cv2.VideoCapture(videoPath)

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
        cv2.imwrite(args["outputdir"] + "/" + videoPath.split(os.path.sep)[-2] + "/" + str(i) + ".jpg",
                    imutils.rotate(cv2.resize(mask, (args["resize"][0], args["resize"][1])), int(args["rotation"])))
        framecount += 1
        i += 1
    cap.release()
    cv2.destroyAllWindows()
