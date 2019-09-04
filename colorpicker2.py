
import cv2
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
points = []
colorsBGR=[]
colorsHSV=[]
upperlimit=([0,0,0])#upperlimits in format [maxHue,maxSaturation,maxValue]
bottomlimit=([179,255,255])#bottomlimits in format [minHue,minSaturation,minValue]

def ginput(event, x, y, flags, param):
# grab references to the global variables
    global points
    global point
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        point=(y,x)
        points.append(point)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (point[1], point[0]),4, (117, 127, 167))
# load the image, clone it, and setup the mouse callback function
image=cv2.imread(r"C:\Users\malfo\Pictures\Camera Roll\1.jpg")
image = cv2.resize(image,(512,512))
cv2.namedWindow("image")
cv2.setMouseCallback("image", ginput)
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF 
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break
#iterate over points and append the color of each pixel in bgr to colorsBGR
#then, we change them to hsv and store them in colorsHSV
for p in points:
    colorsBGR.append(image[p])
    c=np.uint8([[[image[p][0],image[p][1],image[p][2]]]])
    colorsHSV.append(cv2.cvtColor(c,cv2.COLOR_BGR2HSV))
#For each color stored in colorsHSV we check if its components are cointained 
#in our ilimits. If not, limits are updated
for c in colorsHSV:
    #if Hue is bigger than our upper limit, update upper limit
    if c[0][0][0]>upperlimit[0]:
        upperlimit[0]=c[0][0][0]
    #if Hue is smaller than our bottom limit, update bottom limit
    if c[0][0][0]<bottomlimit[0]:
        bottomlimit[0]=c[0][0][0]
        
    #if Saturation is bigger than our upper limit, update upper limit
    if c[0][0][1]>upperlimit[1]:
        upperlimit[1]=c[0][0][1]
    #if Saturation is smaller than our bottom limit, update bottom limit
    if c[0][0][1]<bottomlimit[1]:
        bottomlimit[1]=c[0][0][1]
        
    #if Value is bigger than our upper limit, update upper limit
    if c[0][0][2]>upperlimit[2]:
        upperlimit[2]=c[0][0][2]
    #if Value is smaller than our bottom limit, update bottom limit
    if c[0][0][2]<bottomlimit[2]:
        bottomlimit[2]=c[0][0][2]
# close all open windows
cv2.destroyAllWindows()
print (str(bottomlimit)+ str(upperlimit))