import cv2 as cv
# reading image 

img=cv.imread("niraj.jpg")

if img is not None:
    print("image found")
    print(img.shape) 
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imwrite("newsaved.jpg",gray)
    cv.imshow("photo",img)
    cv.waitKey(0)
    cv.destroyAllWindows() 
else:
    print("Imaage not loaded")

# #reading videos

# capture=cv.VideoCapture("video.mp4")

# while True:
#     isTrue ,frame= capture.read()
#     cv.imshow("Video",frame)

#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyALLWindows()
