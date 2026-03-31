import  cv2 as cv
import mediapipe as mp

class handDetector():

    def __init__(self,mode = False, maxHands = 2, detectionCon = 0.5, trackcon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackcon)

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if hasattr(self, 'results') and hasattr(self.results, 'multi_hand_landmarks') and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw = mp.solutions.drawing_utils
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
        return img
    
    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
        if hasattr(self, 'results') and hasattr(self.results, 'multi_hand_landmarks') and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] 
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)

        return self.lmList
    
    

if __name__ == "__main__":
     cap = cv.VideoCapture(0)
     detector = handDetector()
     while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
     cap.release()
     cv.destroyAllWindows()


        
