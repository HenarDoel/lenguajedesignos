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
ap.add_argument("-r", "--rotation", default="0",required=False,
	help="rotation in degrees")

args = vars(ap.parse_args())

# Opens the Video file
bottomlimit=np.array([130,90,10])
upperlimit=np.array([160,255,255])

i=0

cap= cv2.VideoCapture(args["video"])    

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)
    mask = cv2.inRange(blurred, bottomlimit, upperlimit)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=4)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imwrite(args["outputdir"]+"/"+str(i)+"-"+""+".jpg",imutils.rotate(cv2.resize(mask, (512,512)),int(args["rotation"])))
    i+=1 
cap.release()
cv2.destroyAllWindows()



