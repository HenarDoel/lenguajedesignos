# authors: @henardoel and @marloquemegusta
# usage: this script grabs any videos from the subfolders on the -v directory.
# then, it splits this videos into frames and save them on the subfolders of
# -o directory where the name of the subfolder matches with the name of the
# subfolder on -v

import argparse
import cv2
import imutils
import os

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

args = vars(ap.parse_args())

videoPaths = []
# r=root, d=directories, f = files
# we create an array that contains the paths of all files on -v and subfolders
for r, d, f in os.walk(args["videos"]):
    for file in f:
        videoPaths.append(os.path.join(r, file))

# Opens the Video file
i = 0
framecount = 0
# for each file on videoPaths we split it into frames and proceed to save them
# on the corresponding folder
for videoPath in videoPaths:
    cap = cv2.VideoCapture(videoPath)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % int(args["decimate"]) == 0:
            cv2.imwrite(args["outputdir"] + "/" + videoPath.split(os.path.sep)[-2] + "/" + str(i) + ".jpg",
                        imutils.rotate(cv2.resize(frame, (256, 256)), int(args["rotation"])))
            framecount += 1
        i += 1
    cap.release()
    cv2.destroyAllWindows()
print(framecount)
