def run_mouse():
    import cv2
    import mediapipe as mp
    import pyautogui
    import numpy as np

    # setup
    SMOOTHING = 5  # Higher = smoother but can be laggier
    plocX, plocY = 0, 0  # Previous finger position
    clocX, clocY = 0, 0  # Current finger position

    cap = cv2.VideoCapture(0)  

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    screen_width, screen_height = pyautogui.size()  

    print(" Mouse Active.Press 'q' to quit the window.")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)  
        frame_height, frame_width, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger = hand_landmarks.landmark[8]  
                
                x = int(index_finger.x * frame_width)
                y = int(index_finger.y * frame_height)

                mouse_x = np.interp(x, (0, frame_width), (0, screen_width))
                mouse_y = np.interp(y, (0, frame_height), (0, screen_height))

                clocX = plocX + (mouse_x - plocX) / SMOOTHING
                clocY = plocY + (mouse_y - plocY) / SMOOTHING

                # Move the actual mouse cursor
                pyautogui.moveTo(clocX, clocY)

                plocX, plocY = clocX, clocY  # Update previous 

        cv2.imshow("Mouse Controller", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 

    cap.release()
    cv2.destroyAllWindows()