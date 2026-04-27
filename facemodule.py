import cv2 as cv
import mediapipe as mp

class faceMeshDetector():

    def __init__(self, mode=False, maxFaces=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils


    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_CONTOURS
                    )
        return img


    def findPosition(self, img, faceNo=0, draw=True):
        self.lmList = []
        h, w, c = img.shape

        if self.results.multi_face_landmarks:
            myFace = self.results.multi_face_landmarks[faceNo]

        # Select eye landmarks
        ids = [33, 133, 159, 145, 362, 263, 386, 374]

        for id in ids:
            lm = myFace.landmark[id]
            cx, cy = int(lm.x * w), int(lm.y * h)

            self.lmList.append([id, cx, cy])

            if draw:
                cv.circle(img, (cx, cy), 3, (0, 255, 0), cv.FILLED)

        return self.lmList  
    
    def findEyeCenter(self, lmList):
    # Left eye
     x1, y1 = lmList[0][1], lmList[0][2]  # 33
     x2, y2 = lmList[1][1], lmList[1][2]  # 133

     left_cx = (x1 + x2) // 2
     left_cy = (y1 + y2) // 2

    # Right eye
     x3, y3 = lmList[4][1], lmList[4][2]  # 362
     x4, y4 = lmList[5][1], lmList[5][2]  # 263

     right_cx = (x3 + x4) // 2
     right_cy = (y3 + y4) // 2

     return (left_cx, left_cy), (right_cx, right_cy)


                
                
                
            



    


if __name__ == "__main__":
    import time

    cap = cv.VideoCapture(0)
    detector = faceMeshDetector()

    pTime = 0

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        img = detector.findFaces(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
              left_eye, right_eye = detector.findEyeCenter(lmList)
              cv.circle(img, left_eye, 1, (0, 0, 255), cv.FILLED)
              cv.circle(img, right_eye, 1, (255, 0, 0), cv.FILLED)
          

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f'FPS: {int(fps)}', (10, 50),
                   cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        cv.imshow("Face Mesh", img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()