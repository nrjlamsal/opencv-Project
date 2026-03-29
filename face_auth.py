import cv2 as cv
import numpy as np
import face_recognition as fr

numpyimg = fr.load_image_file("niraj.jpg")

face_locations = fr.face_locations(numpyimg)
know_encoding = fr.face_encodings(numpyimg,face_locations)
face0endcoding= know_encoding[0] # face0endcoding is now not list but a numpy array of 128 dimensions
np.save("niraj_encoding.npy",know_encoding)

vdptr = cv.VideoCapture(0)
while True:
    _,frame = vdptr.read()
    if _ == False:
        break

    new_facelocations= fr.face_locations(frame)
    new_faceencodings = fr.face_encodings(frame,new_facelocations)
    if len(new_faceencodings) == 0:
        continue # if no face is detected in the frame then skip the rest of the loop and continue to the next iteration
    face1encoding = new_faceencodings[0] # face1enoding is now not list but a numpy array of 128 dimensions
    compare = fr.compare_faces([face0endcoding],face1encoding)  # compare faces and face distance needs list of encodings as first argument and single encoding as second argument
    distance = fr.face_distance([face0endcoding],face1encoding)
    break    

 
   

