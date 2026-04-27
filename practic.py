import cv2 as cv
import numpy as np
import handmodule as htm

vdoptr = cv.VideoCapture(0 )
detector = htm.handDetector(detectionCon=0.85)
drawcolor = (0, 0, 255)  # default red
eraser_mode = False
brushThickenss = 15
xp,yp = 0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)



HEADER_H = 100
STATUS_H = 30
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 215, 255),  # Yellow
]
LABELS = ["RED", "GREEN", "BLUE", "YELLOW", "ERASER"]

def draw_header(img):
    h, w, _ = img.shape
    n = 5
    slot_w = w // n

    # Color slots
    bg_colors = [
        (50, 75, 226),   # Red
        (17, 109, 59),   # Green
        (165, 95, 24),   # Blue
        (23, 95, 186),   # Yellow
        (65, 68, 68),    # Eraser gray
    ]
    for i, (bgr, label) in enumerate(zip(bg_colors, LABELS)):
        x1, x2 = i * slot_w, (i + 1) * slot_w
        cv.rectangle(img, (x1, 0), (x2, HEADER_H), bgr, cv.FILLED)

        # Circle preview
        cx = (x1 + x2) // 2
        if i < 4:
            cv.circle(img, (cx, 38), 14, (255, 255, 255), cv.FILLED)
        else:
            # Eraser icon — a small white rect
            cv.rectangle(img, (cx - 14, 28), (cx + 14, 48), (220, 220, 220), cv.FILLED)
            cv.line(img, (cx, 28), (cx, 48), (130, 130, 130), 2)

        # Label
        (tw, _), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv.putText(img, label, (cx - tw // 2, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

    # Active slot indicator (white underline)
    if eraser_mode:
        active = 4
    else:
        active = [(0,0,255),(0,255,0),(255,0,0),(0,215,255)].index(drawcolor) if drawcolor in [(0,0,255),(0,255,0),(255,0,0),(0,215,255)] else 0
    ax1, ax2 = active * slot_w, (active + 1) * slot_w
    cv.rectangle(img, (ax1, HEADER_H - 5), (ax2, HEADER_H), (255, 255, 255), cv.FILLED)

def draw_statusbar(img, mode_text):
    h, w, _ = img.shape
    cv.rectangle(img, (0, h - STATUS_H), (w, h), (20, 20, 20), cv.FILLED)
    cv.putText(img, f"Mode: {mode_text}", (12, h - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
    if not eraser_mode:
        cv.circle(img, (w - 20, h - STATUS_H // 2), 8, drawcolor, cv.FILLED)

while True:
    _, frame = vdoptr.read()
    frame = cv.flip(frame, 1)
    frame = detector.findHands(frame, draw=True)
    lmlist = detector.findPosition(frame, draw=True)
    mode_text = "Idle"

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        fingers = detector.fingersUp()
        h, w, _ = frame.shape
        n = 5
        slot_w = w // n

        if fingers[1] and fingers[2]:
            mode_text = "Selection"
            if y1 < HEADER_H:
                slot = x1 // slot_w
                if slot == 0: drawcolor = (0, 0, 255);  eraser_mode = False
                elif slot == 1: drawcolor = (0, 255, 0);  eraser_mode = False
                elif slot == 2: drawcolor = (255, 0, 0);  eraser_mode = False
                elif slot == 3: drawcolor = (0, 215, 255); eraser_mode = False
                elif slot == 4: eraser_mode = True
            cv.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25),
                         (200, 200, 200) if eraser_mode else drawcolor, cv.FILLED)

        elif fingers[1] and not fingers[2]:
            mode_text = "Eraser" if eraser_mode else "Drawing"
            color = (200, 200, 200) if eraser_mode else drawcolor
            cv.circle(frame, (x1, y1), 15 if not eraser_mode else 25, color, cv.FILLED)
            if xp==0 and yp ==0:
                xp,yp = x1,y1
            cv.line(frame,(xp,yp),(x1,y1),drawcolor,brushThickenss)
            cv.line(imgCanvas,(xp,yp),(x1,y1),drawcolor,brushThickenss)
            xp,yp = x1,y1

    draw_header(frame)
    draw_statusbar(frame, mode_text)

    cv.imshow("AirCanvas", frame)
    cv.imshow( "Canvas", imgCanvas)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vdoptr.release()
cv.destroyAllWindows()