# to use it, run splitter.py -v targetvideo.mp4 -o outputdirectory -r optionalrotation -s resizeheight resizewidth

import argparse
import cv2
import imutils

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

# Opens the Video file
i = 0

cap = cv2.VideoCapture(args["video"])

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(args["outputdir"] + "/" + str(i) + ".jpg",
                imutils.rotate(cv2.resize(frame, (args["resize"][0], args["resize"][1])), int(args["rotation"])))
    i += 1
cap.release()
cv2.destroyAllWindows()
