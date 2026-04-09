import cv2 as cv
import numpy as np
import mediapipe as mp
import handmodule as htm
import time

brushThickenss = 15
ereserThickenss= 50

vdoptr = cv.VideoCapture(0)

dectctor = htm.handDetector(detectionCon = 0.85)
drawcolor = (255, 0, 255)
xp,yp = 0,0
imgCanvas = None

while True:
    _,RBGimg = vdoptr.read()
    RBGimg = cv.flip(RBGimg, 1)
    if imgCanvas is None:
        imgCanvas = np.zeros_like(RBGimg)
    RBGimg = dectctor.findHands(RBGimg,draw = True)
    lmlist = dectctor.findPosition(RBGimg, draw = True)

    if len(lmlist) != 0:
        # this is the tip of index and middle fingers and we will use these to draw on the canvas
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = dectctor.fingersUp()
        # print(fingers)
        h, w, c = RBGimg.shape

# Draw header
        cv.rectangle(RBGimg, (0, 0), (w//4, 125), (0, 0, 255), cv.FILLED)
        cv.rectangle(RBGimg, (w//4, 0), (w//2, 125), (0, 255, 0), cv.FILLED)
        cv.rectangle(RBGimg, (w//2, 0), (3*w//4, 125), (255, 0, 0), cv.FILLED)
        cv.rectangle(RBGimg, (3*w//4, 0), (w, 125), (0, 0, 0), cv.FILLED)


        if  fingers[1] and fingers[2]:
            print("Selection Mode")
            if y1 < 125:
                if 0 < x1 < w//4:
                    print("Color: Red")
                    drawcolor = (0, 0, 255)
                elif w//4 < x1 < w//2:
                    print("Color: Green")
                    drawcolor = (0, 255, 0)
                elif w//2 < x1 < 3*w//4:
                    print("Color: Blue")
                    drawcolor = (255, 0, 0)
                elif 3*w//4 < x1 < w:
                    print("Color: Black")
                    drawcolor = (0, 0, 0 )
            cv.rectangle(RBGimg, (x1, y1-25), (x2, y2+25), drawcolor, cv.FILLED)

        elif fingers[1] and  not    fingers[2]:
            cv.circle(RBGimg, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            print("Drawing Mode")
            if xp==0 and yp ==0:
                xp,yp = x1,y1
            if drawcolor == (0,0,0):
                cv.line(RBGimg,(xp,yp),(x1,y1),drawcolor,ereserThickenss)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawcolor,ereserThickenss)
            else:
                 cv.line(RBGimg,(xp,yp),(x1,y1),drawcolor,brushThickenss)
                 cv.line(imgCanvas,(xp,yp),(x1,y1),drawcolor,brushThickenss)
            xp,yp = x1,y1

    imggray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv = cv.threshold(imggray,58,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    RBGimg = cv.bitwise_and(RBGimg,imgInv)
    RBGimg = cv.bitwise_or(RBGimg,imgCanvas)




    RBGimg = cv.addWeighted(RBGimg,0.5,imgCanvas,0.5,0)
    cv.imshow("AirCanvas", RBGimg)
    cv.imshow( "Canvas", imgCanvas)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    








