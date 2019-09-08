# authors: @henardoel and @marloquemegusta
# usage: this script grabs any videos from the subfolders on the -v directory.
# then, it splits this videos into frames and save them on the subfolders of
# -o directory where the name of the subfolder matches with the name of the
# subfolder on -v. -r corresponds to rotation, -d to decimate and -s to resize.

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
        # if the parameter "d" (decimate) has been passed as an argument, only the frames according to that decimate
        # ratio are saved
        if i % int(args["decimate"]) == 0:
            # the frames are save to disk each one as separated images
            cv2.imwrite(args["outputdir"] + "/" + videoPath.split(os.path.sep)[-2] + "/" + str(i) + ".jpg",
                        imutils.rotate(cv2.resize(frame, (256, 256)), int(args["rotation"])))
            framecount += 1
        i += 1
    cap.release()
    cv2.destroyAllWindows()
print(framecount)
