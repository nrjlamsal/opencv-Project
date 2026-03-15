import cv2 as cv

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml") 
smile_cascade = cv.CascadeClassifier("haarcascade_smile.xml") 


cap = cv.VideoCapture(0)

while True:

    ret,frame = cap.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces= face_cascade.detectMultiScale(gray,1.1,5)

    """
     detechMultiScale() - scan and detect faces
     scalefactor=1.1 balance ,not too slow,blind
     minNeighbough=5    
    """
    for(x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        frame=cv.flip(frame,1)
    
    roi_gray = gray[y:y+h,x:x+h]
    roi_color = frame[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray,1.1,10)
    if len(eyes) > 0:
        cv.putText(frame,"Eyes Detected",(x,y-30),cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),4)

    smiles = smile_cascade.detectMultiScale(roi_gray,1.7,20)
    if len(smiles) > 0:
        cv.putText(frame,"Smile Detected",(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),4)


    cv.imshow("WebCam For Face Dection",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()




