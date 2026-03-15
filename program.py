import cv2 as cv

# img = cv.imread("niraj.jpg")

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# _, thresh = cv.threshold(gray,130,255,cv.THRESH_BINARY)

# contours, hierarchy = cv.findContours(
#         thresh,
#         cv.RETR_EXTERNAL,
#         cv.CHAIN_APPROX_SIMPLE
# )

# cv.drawContours(img, contours, -1, (134,180,255), 2)
# for cnt in contours:

#     perimeter = cv.arcLength(cnt, True)

#     approx = cv.approxPolyDP(cnt, 0.02*perimeter, True)
#     shape=""
#     sides = len(approx)
#     if sides == 3:
#         shape = "Triangle"
#     elif sides == 4:
#         shape = "Rectangle"
#     elif sides > 4:
#         shape = "Circle"
#     cv.drawContours(img,[approx],0,(0,255,0),2)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]-10
#     cv.putText(img,shape,(x,y),cv.FONT_HERSHEY_COMPLEX,)

# cv.imshow("Contours", img)
# cv.waitKey(0)
# cv.destroyAllWindows()




#reading videos

vd=cv.VideoCapture(0)

width=int(vd.get(cv.CAP_PROP_FRAME_WIDTH))
height=int(vd.get(cv.CAP_PROP_FRAME_HEIGHT))
codec=cv.VideoWriter_fourcc(*'XVID')
recorder=cv.VideoWriter("reocrded.mp4",codec,1,(width,height))

while True:
    sucess,frame= vd.read()
    if not sucess:
        break

    frame = cv.flip(frame,1) 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(
        thresh,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
        )
    cv.drawContours(frame, contours, -1, (0,255,0), 2)

    # recorder.write(frame)
    cv.imshow("Recording live",frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vd.release()
# recorder.release()
cv.destroyAllWindows()

     




 