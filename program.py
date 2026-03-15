import cv2 as cv
import numpy as np
# reading image 

img=cv.imread("niraj.jpg")

sharpen_kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

if img is None:
    print("No image is found")
else:
    print(img.shape)
    y,x=img.shape[:2]
  
    sharpened = cv.filter2D(img,-1,sharpen_kernel)
     
    cv.imshow("Normal",img)
    cv.imshow("Sharpaned ",sharpened)

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

     




 