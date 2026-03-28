import cv2 as cv
import numpy as np
import face_recognition as fr

numpyimg = fr.load_image_file("niraj.jpg")

face_locations = fr.face_locations(numpyimg)
know_encoding = fr.face_encodings(numpyimg,face_locations)
face0endcoding= know_encoding[0]
np.save("niraj_encoding.npy",know_encoding)

vdptr = cv.VideoCapture(0)
while True:
    _,frame = vdptr.read()
    if _ == False:
        break

    new_facelocations= fr.face_locations(frame)
    new_faceencodings = fr.face_encodings(frame,new_facelocations)
    face1encoding = new_faceencodings[0]
    compare = fr.compare_faces([face0endcoding],face1encoding)    
    distance = fr.face_distance([face0endcoding],face1encoding)    

    print(compare)
    print(distance)
    if distance < 0.5:
        print("niraj found using distance")

    if compare[0] == True:
        print("niraj found using compare")
    break

