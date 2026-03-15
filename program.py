import cv2 as cv
# reading image 

img=cv.imread("niraj.jpg")

if img is None:
    print("No image is found")
else:
    print(img.shape)
    y,x=img.shape[:2]
    blurred= cv.GaussianBlur(img,(x,y),10)
     
    cv.imshow("Noraml Image ",img)
    cv.imshow("Blured ",blurred)
    cv.waitKey(0)
    cv.destroyAllWindows()





# #reading videos

# vd=cv.VideoCapture(0)

# width=int(vd.get(cv.CAP_PROP_FRAME_WIDTH))
# height=int(vd.get(cv.CAP_PROP_FRAME_HEIGHT))
# codec=cv.VideoWriter_fourcc(*'XVID')
# recorder=cv.VideoWriter("reocrded.mp4",codec,1,(width,height))

# while True:
#     sucess,frame= vd.read()
#     if not sucess:
#         break
#     frame = cv.flip(frame,1) 
#     recorder.write(frame)
#     cv.imshow("Recording live",frame)
    
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# vd.release()
# recorder.release()
# cv.destroyAllWindows()

     




 