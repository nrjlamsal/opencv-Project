import cv2 as cv
# reading image 

img=cv.imread("niraj.jpg")

if img is None:
    print("No image is found")
else:
    print(img.shape)
    croppedimg=img[0:959,0:719]
    cv.imshow("Photo1",img)

    cv.imshow("Photo",croppedimg)
    cv.waitKey(0)
    cv.destroyAllWindows()





# #reading videos

# capture=cv.VideoCapture("video.mp4")

# while True:
#     isTrue ,frame= capture.read()
#     cv.imshow("Video",frame)
#heloo

#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyALLWindows()
