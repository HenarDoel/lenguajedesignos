import cv2
import numpy as np
import tensorflow as tf


class SimplePreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def maskExtraction(self, frame, bottomLimit, upperLimit):
        # initialize the mask with the size previously defined
        maskRGB = np.zeros([self.height, self.width, 3])

        # frame is converted to hsv for easier hue management
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # hsv frame is blurred so high frequency noise is reduced
        blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

        # pixel wise mask is created according to color thresholds
        mask = cv2.inRange(blurred, bottomLimit, upperLimit)

        # mask is processed to reduce noise and close little holes
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=4)

        # the mask is processed so it has the right shape to be introduced
        # to our trained network to be classified
        mask = cv2.resize(mask, (self.height, self.width))
        mask = mask.astype("float") / 255.0
        maskRGB[:, :, 0] = mask
        maskRGB[:, :, 1] = mask
        maskRGB[:, :, 2] = mask

        return maskRGB
