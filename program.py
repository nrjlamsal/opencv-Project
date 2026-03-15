import cv2
import numpy as np

# Create two blank black images
img1 = np.zeros((300, 300), dtype="uint8")
img2 = np.zeros((300, 300), dtype="uint8")

# Draw a white circle on the first image
cv2.circle(img1, (150, 150), 100, 255, -1)

# Draw a white rectangle on the second image
cv2.rectangle(img2, (100, 100), (250, 250), 255, -1)

# Perform bitwise operations
bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_not = cv2.bitwise_not(img1)

# Display the results
cv2.imshow("Circle", img1)
cv2.imshow("Rectangle", img2)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("NOT", bitwise_not)

cv2.waitKey(0)
cv2.destroyAllWindows()




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

     




 