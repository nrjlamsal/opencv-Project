

import cv2 as cv
import numpy as np
import mediapipe as mp
import sys
import time
import math


from face_auth import run_auth

try:
    from canvas import run_canvas
    CANVAS_AVAILABLE = True
except ImportError as e:
    CANVAS_AVAILABLE = False
    print(f"[WARN] Air Canvas not available: {e}")

try:
    from handkeyboard import run_keyboard
    KEYBOARD_AVAILABLE = True
except ImportError as e:
    KEYBOARD_AVAILABLE = False
    print(f"[WARN] Hand Keyboard not available: {e}")

try:
    from blink_morse import run_morse
    MORSE_AVAILABLE = True
except ImportError as e:
    MORSE_AVAILABLE = False
    print(f"[WARN] Blink Morse not available: {e}")

try:
    from rps_game import run_rps
    RPS_AVAILABLE = True
except ImportError as e:
    RPS_AVAILABLE = False
    print(f"[WARN] Rock Paper Scissors not available: {e}")

try:
    from hand_mouse import run_mouse
    MOUSE_AVAILABLE = True
except ImportError as e:
    MOUSE_AVAILABLE = False
    print(f"[WARN] Hand Mouse not available: {e}")

try:
    from face_puzzle import run_puzzle
    PUZZLE_AVAILABLE = True
except ImportError as e:
    PUZZLE_AVAILABLE = False
    print(f"[WARN] Face Puzzle not available: {e}")



mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

# menu configuration

WINDOW_W = 1050
WINDOW_H = 680

TILES = [
    {
        "label": "Air Canvas",
        "desc": "Draw with your hand",
        "color": (180, 60, 60),
        "key": "1",
        "available": True,          
    },
    {
        "label": "Hand Keyboard",
        "desc": "Type with finger dwell",
        "color": (60, 140, 60),
        "key": "2",
        "available": True,
    },
    {
        "label": "Blink Morse",
        "desc": "Communicate via blinks",
        "color": (60, 60, 180),
        "key": "3",
        "available": True,
    },
    {
        "label": "Rock Paper Scissors",
        "desc": "Gesture-based RPS game",
        "color": (170, 80, 180),
        "key": "4",
        "available": True,
    },
    {
        "label": "Hand Mouse",
        "desc": "Control cursor by hand",
        "color": (60, 150, 170),
        "key": "5",
        "available": True,
    },
    {
        "label": "Face Puzzle",
        "desc": "Sliding puzzle with face",
        "color": (170, 130, 50),
        "key": "6",
        "available": True,
    },
]

TILES[0]["available"] = CANVAS_AVAILABLE
TILES[1]["available"] = KEYBOARD_AVAILABLE
TILES[2]["available"] = MORSE_AVAILABLE
TILES[3]["available"] = RPS_AVAILABLE
TILES[4]["available"] = MOUSE_AVAILABLE
TILES[5]["available"] = PUZZLE_AVAILABLE

# Layout 
COLS = 3
ROWS = 2
TILE_W = 250
TILE_H = 180
TILE_GAP_X = 35
TILE_GAP_Y = 30

DWELL_TIME = 1.2        
PINCH_THRESH = 40        


VERSION = "1.0.0"




def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    """Draw a rectangle with rounded corner"""
    overlay = img.copy()
    cv.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv.circle(overlay, (x + r, y + r), r, color, -1)
    cv.circle(overlay, (x + w - r, y + r), r, color, -1)
    cv.circle(overlay, (x + r, y + h - r), r, color, -1)
    cv.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    if alpha < 1.0:
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        np.copyto(img, overlay)


def draw_progress_ring(img, cx, cy, radius, progress, color, thickness=3):
    """Draw a circular progress arc  around a center point"""
    angle = int(360 * progress)
    if angle > 0:
        cv.ellipse(img, (cx, cy), (radius, radius), -90, 0, angle, color, thickness)


def get_tile_positions():
    total_w = COLS * TILE_W + (COLS - 1) * TILE_GAP_X
    total_h = ROWS * TILE_H + (ROWS - 1) * TILE_GAP_Y
    start_x = (WINDOW_W - total_w) // 2
    start_y = (WINDOW_H - total_h) // 2 + 30 
    positions = []
    for idx in range(len(TILES)):
        row = idx // COLS
        col = idx % COLS
        x = start_x + col * (TILE_W + TILE_GAP_X)
        y = start_y + row * (TILE_H + TILE_GAP_Y)
        positions.append((x, y))
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
    cv.putText(canvas, name,(50, 160), cv.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)
    cv.putText(canvas, "Module Not Available",(50, 220), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 1)
    cv.putText(canvas, "Press any key to go back",(50, 300), cv.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    cv.imshow(name, canvas)
    cv.waitKey(0)
    cv.destroyWindow(name)


def open_camera(index=0, width=640, height=480, fps=60):
    
    for attempt in range(3):
        cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv.CAP_PROP_FPS, fps)
            return cap
        print(f"  [Camera] Attempt {attempt + 1}/3 failed, retrying...")
        time.sleep(0.5)
    print("  [Camera] ERROR: Could not open camera after 3 attempts.")
    return None



def show_menu(cap):
  
    tile_positions = get_tile_positions()

    # Dwell state
    dwell_tile = -1          
    dwell_start = 0.0        
    dwell_progress = 0.0    

    # Camera size
    CAM_W = 200
    CAM_H = 150

    hovered_tile = -1

    two_hand_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands_model.process(rgb)

        # Two-hand exit: show both hands for 0.5s
        if (result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — quitting application.")
                return 0
        else:
            two_hand_count = 0

        now = time.time()

        canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)

        #  Title 
        title = "Contactless HCI System"
        (tw, _), _ = cv.getTextSize(title, cv.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = (WINDOW_W - tw) // 2
        cv.putText(canvas, title, (tx, 50),cv.FONT_HERSHEY_DUPLEX, 1.0, (220, 220, 230), 2, cv.LINE_AA)


        subtitle = "Point at a tile  |  Pinch & hold to select  |  Keys 1-6"
        (sw, _), _ = cv.getTextSize(subtitle, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        sx = (WINDOW_W - sw) // 2
        cv.putText(canvas, subtitle, (sx, 78), cv.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 130), 1, cv.LINE_AA)

        hovered_tile = -1
        is_pinching = False
        cursor_pos = None

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            ix = int(lm[8].x * WINDOW_W)
            iy = int(lm[8].y * WINDOW_H)
            cursor_pos = (ix, iy)

            pinch_dist, mx, my = get_pinch_distance(lm, WINDOW_W, WINDOW_H)
            is_pinching = pinch_dist < PINCH_THRESH

            for i, (tile_x, tile_y) in enumerate(tile_positions):
                if point_in_tile(ix, iy, tile_x, tile_y):
                    hovered_tile = i
                    break

            # dwell logic
            if is_pinching and hovered_tile != -1:
                if dwell_tile == hovered_tile:
                    elapsed = now - dwell_start
                    dwell_progress = min(elapsed / DWELL_TIME, 1.0)

                    if dwell_progress >= 1.0:
                        cv.destroyAllWindows()
                        cap.release()
                        return hovered_tile + 1
                else:
                    dwell_tile = hovered_tile
                    dwell_start = now
                    dwell_progress = 0.0
            elif is_pinching and hovered_tile == -1:
                dwell_tile = -1
                dwell_progress = 0.0
            else:
                dwell_tile = -1
                dwell_progress = 0.0

        else:
            dwell_tile = -1
            dwell_progress = 0.0

        for i, (tile, (tile_x, tile_y)) in enumerate(zip(TILES, tile_positions)):
            is_hovered = (i == hovered_tile)
            is_dwelling = (i == dwell_tile and dwell_progress > 0)
            is_available = tile["available"]

            r, g, b = tile["color"]
            if not is_available:
                r, g, b = r // 3, g // 3, b // 3

            if is_hovered:
                color = (min(r + 50, 255), min(g + 50, 255), min(b + 50, 255))
            else:
                color = (r, g, b)

            draw_rounded_rect(canvas, tile_x, tile_y, TILE_W, TILE_H, 14, color, alpha=0.9)

            if is_hovered:
                pulse = abs(math.sin(now * 3))
                glow = int(160 + 95 * pulse)
                cv.rectangle(canvas,(tile_x - 1, tile_y - 1),(tile_x + TILE_W + 1, tile_y + TILE_H + 1),(glow, glow, glow), 2)

            if is_dwelling:
                ring_cx = tile_x + TILE_W // 2
                ring_cy = tile_y + TILE_H // 2
                ring_color = (0, int(200 * dwell_progress), 255)
                draw_progress_ring(canvas, ring_cx, ring_cy, 38, dwell_progress, ring_color, thickness=4)
                pct_text = f"{int(dwell_progress * 100)}%"
                (pw, _), _ = cv.getTextSize(pct_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.putText(canvas, pct_text,(ring_cx - pw // 2, ring_cy + 6),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            badge_cx = tile_x + TILE_W - 28
            badge_cy = tile_y + 28
            cv.circle(canvas, (badge_cx, badge_cy), 18, (255, 255, 255), -1)
            cv.putText(canvas, tile["key"],(badge_cx - 7, badge_cy + 7),cv.FONT_HERSHEY_DUPLEX, 0.7, (r, g, b), 2)

            cv.putText(canvas, tile["label"],(tile_x + 16, tile_y + TILE_H - 52),cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

            cv.putText(canvas, tile["desc"],(tile_x + 16, tile_y + TILE_H - 28),cv.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 210), 1, cv.LINE_AA)

            if is_hovered and not is_dwelling:
                cv.putText(canvas, "PINCH TO SELECT",(tile_x + 16, tile_y + TILE_H - 8),cv.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

            if is_dwelling:
                cv.putText(canvas, "SELECTING...",(tile_x + 16, tile_y + TILE_H - 8),cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1)

            if not is_available:
                cv.putText(canvas, "UNAVAILABLE",(tile_x + TILE_W // 2 - 55, tile_y + TILE_H // 2 + 5),cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if cursor_pos:
            ix, iy = cursor_pos
            cv.circle(canvas, (ix, iy), 10, (255, 255, 255), 2)
            cv.circle(canvas, (ix, iy), 3, (255, 255, 255), -1)

            if is_pinching:
                cv.circle(canvas, (ix, iy), 18, (0, 255, 255), 2)

        cam_small = cv.resize(frame, (CAM_W, CAM_H))
        xo = WINDOW_W - CAM_W - 15
        yo = WINDOW_H - CAM_H - 40
        cv.rectangle(canvas,(xo - 2, yo - 2),(xo + CAM_W + 2, yo + CAM_H + 2),(60, 60, 70), 1)
        canvas[yo:yo + CAM_H, xo:xo + CAM_W] = cam_small

        footer = f"Press 1-6 to select  |  Q to quit  |  v{VERSION}"
        (fw_t, _), _ = cv.getTextSize(footer, cv.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        fx = (WINDOW_W - fw_t) // 2
        cv.putText(canvas, footer,(fx, WINDOW_H - 12),cv.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 80), 1)

        cv.imshow("HCI System", canvas)

        key = cv.waitKey(1) & 0xFF
        if key == ord('1'):
            cv.destroyAllWindows()
            cap.release()
            return 1
        elif key == ord('2'):
            cv.destroyAllWindows()
            cap.release()
            return 2
        elif key == ord('3'):
            cv.destroyAllWindows()
            cap.release()
            return 3
        elif key == ord('4'):
            cv.destroyAllWindows()
            cap.release()
            return 4
        elif key == ord('5'):
            cv.destroyAllWindows()
            cap.release()
            return 5
        elif key == ord('6'):
            cv.destroyAllWindows()
            cap.release()
            return 6
        elif key == ord('q') or key == 27:
            cv.destroyAllWindows()
            cap.release()
            return 0



FEATURE_MAP = {
    1: ("Air Canvas",           CANVAS_AVAILABLE,   lambda: run_canvas()),
    2: ("Hand Keyboard",        KEYBOARD_AVAILABLE, lambda: run_keyboard()),
    3: ("Blink Morse",          MORSE_AVAILABLE,    lambda: run_morse()),
    4: ("Rock Paper Scissors",  RPS_AVAILABLE,      lambda: run_rps()),
    5: ("Hand Mouse",           MOUSE_AVAILABLE,    lambda: run_mouse()),
    6: ("Face Puzzle",          PUZZLE_AVAILABLE,    lambda: run_puzzle()),
}



def main():
  
    print("=" * 58)
    print("  Contactless HCI System")
    print("  University of Idaho — Programming Languages, Spring 2026")
    print("=" * 58)

    print("\n[STEP 1] Face authentication...")
    authenticated = run_auth()

    if not authenticated:
        print("Access denied. Exiting.")
        sys.exit(0)

    print("Welcome! Loading menu...\n")

    cap = open_camera()
    if cap is None:
        print("FATAL: Cannot open camera. Exiting.")
        sys.exit(1)

    while True:
        choice = show_menu(cap)

        if choice == 0:
            print("Goodbye.")
            break

        name, available, launcher = FEATURE_MAP.get(choice, (None, False, None))

        if name is None:
            continue

        print(f"\n{'─' * 40}")
        print(f"  [MODE] {name}")
        print(f"{'─' * 40}")

        if available:
            try:
                launcher()
            except Exception as e:
                print(f"  [ERROR] {name} crashed: {e}")
                import traceback
                traceback.print_exc()
        else:
            coming_soon(name)

        print(f"\nReturning to menu...")

        cap = open_camera()
        if cap is None:
            print("FATAL: Cannot re-open camera after module exit. Exiting.")
            break

    if cap is not None and cap.isOpened():
        cap.release()
    cv.destroyAllWindows()
    print("System closed.")


if __name__ == "__main__":
    main()