import cv2 as cv
import numpy as np
import mediapipe as mp
import handmodule as htm
import time

vdoptr = cv.VideoCapture(0)
dectctor = htm.handDetector(detectionCon = 0.85)


while True:
    _,RBGimg = vdoptr.read()
    RBGimg = cv.flip(RBGimg, 1)
    RBGimg = dectctor.findHands(RBGimg,draw = True)
    lmlist = dectctor.findPosition(RBGimg, draw = True)
    if len(lmlist) != 0:
        # this is the tip of index and middle fingers and we will use these to draw on the canvas
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = dectctor.fingersUp()
        print(fingers)



    cv.imshow("AirCanvas", RBGimg)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    








