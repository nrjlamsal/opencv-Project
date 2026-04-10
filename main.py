import cv2 as cv
import numpy as np
import mediapipe as mp
import sys

# ─── IMPORTS ──────────────────────────────────────────────
from face_auth import run_auth
try:
    from air_canvas import run_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False
    print("Warning: air_canvas.py not found")

try:
    from eye_keyboard import run_keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

try:
    from hand_mouse import run_mouse
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False

# ─── MEDIAPIPE ────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
WINDOW_W = 900
WINDOW_H = 550

TILES = [
    {"label": "Air Canvas",        "desc": "Draw with your hand",    "color": (180, 60,  60),  " key": "1"},
    {"label": "Eye Keyboard",      "desc": "Type with your eyes",    "color": (60,  140, 60),  " key": "2"},
    {"label": "Hand Mouse", "desc": "Control cursor with hand", "color": (60,  60,  180), "key": "3"},
]

TILE_W        = 210
TILE_H        = 230
TILE_GAP      = 40
PINCH_THRESH  = 40


def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    overlay = img.copy()
    cv.rectangle(overlay, (x + r, y),     (x + w - r, y + h),     color, -1)
    cv.rectangle(overlay, (x, y + r),     (x + w,     y + h - r), color, -1)
    cv.circle(overlay, (x + r,     y + r),     r, color, -1)
    cv.circle(overlay, (x + w - r, y + r),     r, color, -1)
    cv.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def get_tile_positions():
    total_w = len(TILES) * TILE_W + (len(TILES) - 1) * TILE_GAP
    start_x = (WINDOW_W - total_w) // 2
    start_y = (WINDOW_H - TILE_H) // 2
    positions = []
    for i in range(len(TILES)):
        x = start_x + i * (TILE_W + TILE_GAP)
        positions.append((x, start_y))
    return positions


def point_in_tile(px, py, tx, ty):
    return tx < px < tx + TILE_W and ty < py < ty + TILE_H


def get_pinch_distance(lm, w, h):
    tx = int(lm[4].x * w)
    ty = int(lm[4].y * h)
    ix = int(lm[8].x * w)
    iy = int(lm[8].y * h)
    return np.hypot(tx - ix, ty - iy), (tx + ix) // 2, (ty + iy) // 2


def coming_soon(name):
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    cv.putText(canvas, name,
               (50, 160), cv.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)
    cv.putText(canvas, "Coming Soon",
               (50, 220), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 1)
    cv.putText(canvas, "Press any key to go back",
               (50, 300), cv.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    cv.imshow(name, canvas)
    cv.waitKey(0)
    cv.destroyWindow(name)


# ─── MENU ─────────────────────────────────────────────────
def show_menu(cap):
    """
    Shows menu with hand tracking.
    Point index finger at tile to highlight.
    Pinch to confirm selection.
    Returns 1, 2, 3 or 0 to quit.
    """
    tile_positions = get_tile_positions()
    hovered_tile   = -1    
    pinch_tile     = -1   

    CAM_W = 200
    CAM_H = 150

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands_model.process(rgb)

        canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv.putText(canvas, "Contactless HCI System",(WINDOW_W // 2 - 220, 55),cv.FONT_HERSHEY_DUPLEX, 0.9, (220, 220, 220), 1)
        cv.putText(canvas, "Point at a tile  —  Pinch to select",(WINDOW_W // 2 - 185, 88),cv.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        hovered_tile = -1

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            ix = int(lm[8].x * WINDOW_W)
            iy = int(lm[8].y * WINDOW_H)

            pinch_dist, mx, my = get_pinch_distance(lm, WINDOW_W, WINDOW_H)
            is_pinching = pinch_dist < PINCH_THRESH

            for i, (tx, ty) in enumerate(tile_positions):
                if point_in_tile(ix, iy, tx, ty):
                    hovered_tile = i
                    break

            if is_pinching and hovered_tile != -1:
                cv.destroyAllWindows()
                cap.release()
                return hovered_tile + 1

            cv.circle(canvas, (ix, iy), 10, (255, 255, 255), 2)
            cv.circle(canvas, (ix, iy), 3,  (255, 255, 255), -1)

            if is_pinching:
                cv.circle(canvas, (mx, my), 15, (0, 255, 255), -1)

        for i, (tile, (tx, ty)) in enumerate(zip(TILES, tile_positions)):
            is_hovered = i == hovered_tile

            r, g, b   = tile["color"]
            color     = (min(r + 60, 255), min(g + 60, 255), min(b + 60, 255)) \
                        if is_hovered else tile["color"]

            draw_rounded_rect(canvas, tx, ty, TILE_W, TILE_H, 12, color)

            if is_hovered:
                cv.rectangle(canvas, (tx, ty),(tx + TILE_W, ty + TILE_H),(255, 255, 255), 2)
                cv.putText(canvas, "PINCH TO SELECT",(tx + 18, ty + TILE_H - 18), cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            # key badge
            cv.circle(canvas, (tx + TILE_W - 25, ty + 25), 18, (255, 255, 255), -1)
            cv.putText(canvas, tile["key"],(tx + TILE_W - 32, ty + 32),cv.FONT_HERSHEY_DUPLEX, 0.7, tile["color"], 2)

            # label
            cv.putText(canvas, tile["label"],(tx + 16, ty + TILE_H - 65),cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 
            cv.putText(canvas, tile["desc"],(tx + 16, ty + TILE_H - 40),cv.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cam_small = cv.resize(frame, (CAM_W, CAM_H))
        xo = WINDOW_W - CAM_W - 10
        yo = WINDOW_H - CAM_H - 10
        cv.rectangle(canvas, (xo - 2, yo - 2),(xo + CAM_W + 2, yo + CAM_H + 2),(80, 80, 80), 1)
        canvas[yo:yo + CAM_H, xo:xo + CAM_W] = cam_small
        # footer
        cv.putText(canvas, "Press 1 / 2 / 3 to select with keyboard and  Q to quit",(WINDOW_W // 2 - 240, WINDOW_H - 12),cv.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        cv.imshow("HCI System", canvas)

        key = cv.waitKey(1) & 0xFF
        if key == ord('1'):
            cv.destroyAllWindows()
            return 1
        elif key == ord('2'):
            cv.destroyAllWindows()
            return 2
        elif key == ord('3'):
            cv.destroyAllWindows()
            return 3
        elif key == ord('q') or key == 27:
            cv.destroyAllWindows()
            return 0


# ─── MAIN ─────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Contactless HCI System")
    print("=" * 50)

    # step 1 — face auth
    print("\n[STEP 1] Face authentication...")
    authenticated = run_auth()

    if not authenticated:
        print("Access denied. Exiting.")
        sys.exit(0)

    print("Welcome! Loading menu...")

    # open camera for menu
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 60)

    # step 2 — menu loop
    while True:
        choice = show_menu(cap)

        if choice == 0:
            print("Goodbye.")
            break

        elif choice == 1:
            print("\n[MODE] Air Canvas")
            cap.release()                  
            if CANVAS_AVAILABLE:
                run_canvas()
            else:
                coming_soon("Air Canvas")
            cap = cv.VideoCapture(0, cv.CAP_DSHOW)   
            cap.set(cv.CAP_PROP_FPS, 60)

        elif choice == 2:
            print("\n[MODE] Eye Keyboard")
            cap.release()
            if KEYBOARD_AVAILABLE:
                run_keyboard()
            else:
                coming_soon("Eye Keyboard")
            cap = cv.VideoCapture(0, cv.CAP_DSHOW)
            cap.set(cv.CAP_PROP_FPS, 60)

        elif choice == 3:
            print("\n[MODE] Hand Mouse")
            cap.release()
            if MOUSE_AVAILABLE:
                run_mouse()
            else:
                coming_soon("Hand Mouse")
            cap = cv.VideoCapture(0, cv.CAP_DSHOW)
            cap.set(cv.CAP_PROP_FPS, 60)

        print("\nReturning to menu...")

    cap.release()
    cv.destroyAllWindows()
    print("System closed.")


if __name__ == "__main__":
    main()