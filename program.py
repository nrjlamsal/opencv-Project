import cv2 as cv
# reading image 

# img=cv.imread("niraj.jpg")

# if img is None:
#     print("No image is found")
# else:
#     print(img.shape)
#     y,x=img.shape[:2]
#     cv.putText(img," THis is Niraj",(x//2,y//2),cv.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
    
#     cv.imshow("Rotated Image ",img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()





# #reading videos

vd=cv.VideoCapture(0)

while True:
   check,frame=vd.read()
   if  not check :
      print(" Could not read frames")
      break
    
   cv.imshow("Webcam",frame)

   if cv.waitKey(1) & 0xFF==ord('q'):
      print("qutting")
      break
   
vd.release()
cv.destroyAllWindows()

 