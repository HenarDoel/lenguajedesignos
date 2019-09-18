# authors: @henardoel and @marloquemegusta
# usage: run  splitter-colormask.py -v targetvideo.mp4 -o targetdirectory -r rotation -s targetHeight targetWidth
# the input parameter -v could be only one video or a folder that contains videos -v folder_with_videos
import argparse
import cv2
import imutils
import numpy as np
from SimplePreprocessor import SimplePreprocessor
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video")
ap.add_argument("-o", "--outputdir", required=True,
                help="path of the output directory")
ap.add_argument("-r", "--rotation", default="0", required=False,
                help="rotation in degrees")
ap.add_argument("-s", "--resize", default=(96, 96), required=False, nargs=2,
                help="size of frames in case you want to resize them")
args = vars(ap.parse_args())

# set color thresholds
bottomLimit = np.array([130, 90, 10])
upperLimit = np.array([160, 255, 255])

# initialize index for image counting and object Simple Preprocessor from
# which we are going to import some functions
i = 0
sp = SimplePreprocessor(int(args["resize"][0]), int(args["resize"][1]))

# an array that contains the paths of all files on -v and subfolders is created
# r=root, d=directories, f = files
videoPaths = []
for r, d, f in os.walk(args["video"]):
    for file in f:
        videoPaths.append(os.path.join(r, file))

# if videoPaths is empty it means that the parameter -v was a video, not a folder, so we add it to
# the list manually
if len(videoPaths) == 0:
    videoPaths = [args["video"]]


# main loop
for videoPath in videoPaths:

    # Opens the Video file
    cap = cv2.VideoCapture(videoPath)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # we call the Simple Preprocessor function "maskExtraction"
        maskRGB = sp.maskExtraction(frame, bottomLimit, upperLimit)

        # the mask is saved to disk with two methods depending on the input parameter -v
        # if -v was a folder containing videos:
        if len(videoPaths) > 1:
            cv2.imwrite(args["outputdir"] + "/" + videoPath.split(os.path.sep)[-2] + "/" + str(i) + ".jpg",
                        imutils.rotate(cv2.resize(maskRGB, (args["resize"][0], args["resize"][1])),
                                       int(args["rotation"])))
        # if -v was a video:
        else:
            cv2.imwrite(args["outputdir"] + "/" + str(i) + ".jpg",
                        imutils.rotate(cv2.resize(maskRGB, (int(args["resize"][0]), int(args["resize"][1]))),
                                       int(args["rotation"])))
        i += 1

    cap.release()
    cv2.destroyAllWindows()
