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

            # cx, cy = int(l_outer_corner.x * w), int(l_inner_corner.y * h)
            # cv.circle(img, (cx, cy), 1, (0, 255, 0), cv.FILLED)
            l_outer_corner = myFace[33]
            l_inner_corner = myFace[133]
            l_lidregion = myFace[145]
            l_eyebrowregion = myFace[7]
            r_outer_corner = myFace[362]
            r_inner_corner = myFace[263]
            r_lidregion = myFace[386]
            r_eyebrowregion = myFace[374]

            self.lmList.append(l_outer_corner)
            self.lmList.append(l_inner_corner)
            self.lmList.append(l_lidregion)
            self.lmList.append(l_eyebrowregion)
            self.lmList.append(r_outer_corner)
            self.lmList.append(r_inner_corner)
            self.lmList.append(r_lidregion)
            self.lmList.append(r_eyebrowregion)
            
            for (lx,ly) in self.lmList:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(img, (cx, cy), 1, (0, 255, 0), cv.FILLED)
            
            return self.lmlist

             
                
                
                
            



        #     for id, lm in enumerate(myFace.landmark):
        #         h, w, c = img.shape
        #         cx, cy = int(lm.x * w), int(lm.y * h)

        #         self.lmList.append([id, cx, cy])

        #         if draw:
        #             cv.circle(img, (cx, cy), 1, (0, 255, 0), cv.FILLED)

        # return self.lmList


# 🔹 Testing (same style as your hand module)
if __name__ == "__main__":
    import time

    cap = cv.VideoCapture(0)
    detector = faceMeshDetector()

    pTime = 0

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        img = detector.findFaces(img, draw=False)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            print(lmList[1])  # example: print 1 landmark

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