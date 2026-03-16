import cv2 as cv
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv.flip(frame,1)
    frame_height,frame_width = frame.shape[:2]

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
         # Loop over faces
        face1= results.multi_face_landmarks[0]

        liris=face1.landmark[468]
        x = int(liris.x * frame_width)
        y = int(liris.y * frame_height)     
        cv.circle(frame, (x, y), 1, (0,255,0), -1)

        riris=face1.landmark[473]
        x = int(riris.x * frame_width)
        y = int(riris.y * frame_height)     
        cv.circle(frame, (x, y), 1, (0,255,0), -1)

    cv.imshow("Iris Tracking", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()