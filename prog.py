import cv2 as cv
import mediapipe as mp

# Load FaceMesh
# refine_landmarks=True gives us the IRIS points (needed for eye tracking)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # FaceMesh needs RGB, but OpenCV gives BGR — so we convert
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame — this gives us all 468 landmarks
    result = face_mesh.process(rgb)

    # Check if any face was detected
    if result.multi_face_landmarks:
        print(result.multi_face_landmarks)
        print("Face detected!")

    else:
        print("No face detected")

    cv.imshow("FaceMesh", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()