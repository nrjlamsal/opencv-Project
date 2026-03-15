import cv2 as cv
# reading image 

img=cv.imread("niraj.jpg")

if img is None:
    print("No image is found")
else:
    print(img.shape)
    y,x=img.shape[:2]
    cv.circle(img,(x//2,y//2),100,(255,0,0),2)
    
    cv.imshow("Rotated Image ",img)
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
