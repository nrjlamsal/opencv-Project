import cv2 as cv
import numpy as np
# reading image 

img=cv.imread("niraj.jpg",cv.IMREAD_GRAYSCALE)


if img is None:
    print("No image is found")
else:
    print(img.shape)
    y,x=img.shape[:2]
  
    edge=cv.Canny(img,50,150)
    cv.imshow("Normal",img)
    cv.imshow("Canny  ",edge)

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

     




 