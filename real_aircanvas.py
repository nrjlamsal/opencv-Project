"""
🎨 Air Canvas 
  - Deque-based point storage for gap-free strokes

Gesture Controls:
    Index finger ONLY up    -> DRAW mode (index tip = pen)
    Index + Middle fingers   -> SELECT mode (toolbar)
    Everything else          -> PEN LIFT
    Both hands (~0.5s)       -> Return to menu

Keyboard Shortcuts:
    c = clear canvas
    z / y = undo / redo
    +  / - = increase / decrease brush size
    q / ESC = quit
"""

import cv2 as cv
import numpy as np
import handmodule as htm
import time
import colorsys
import math
from collections import deque

def get_rainbow_color():
    h = (time.time() * 0.4) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def finger_distance(lmlist, a, b):
    """Pixel distance between two landmarks"""
    return math.hypot(lmlist[a][1] - lmlist[b][1],lmlist[a][2] - lmlist[b][2])


# rendering strokes

def render_strokes(canvas, strokes):
    """"
    Render strokes onto a canvas by drawing lines between consecutive points.
    """
    for stroke in strokes:
        pts = stroke["points"]
        color = stroke["color"]
        thick = stroke["thickness"]

        for k in range(1, len(pts)):
            if pts[k - 1] is None or pts[k] is None:
                continue
            cv.line(canvas, pts[k - 1], pts[k], color, thick, cv.LINE_AA)


def bake_stroke_to_canvas(img_canvas, stroke):
    """Permanently render a completed stroke onto the canvas."""
    pts = stroke["points"]
    color = stroke["color"]
    thick = stroke["thickness"]
    for k in range(1, len(pts)):
        if pts[k - 1] is None or pts[k] is None:
            continue
        cv.line(img_canvas, pts[k - 1], pts[k], color, thick, cv.LINE_AA)


def composite_canvas(frame, img_canvas, current_stroke=None):
    combined = img_canvas.copy()
    if current_stroke and len(current_stroke["points"]) > 1:
        render_strokes(combined, [current_stroke])
    mask = combined.any(axis=2)
    result = frame.copy()
    result[mask] = combined[mask]
    return result


#  main driver function

def run_canvas():
    """
    Launch the Air Canvas drawing application.
    """
    eraserThickness = 50

    # gesture buffer
    BUFFER_ENTER_DRAW = 3
    BUFFER_ENTER_SEL  = 3
    pending_mode = "LIFT"
    buffer_count = 0

    POST_LIFT_FRAMES = 1
    lift_cooldown = 0

    #Stroke storage
    imgCanvas = None
    current_stroke = None
    canvas_history = []
    redo_stack = []
    MAX_UNDO = 20

    state = {
        "drawcolor": (0, 0, 255),
        "brushThickness": 5,
        "rainbow": True,
        "mode": "LIFT",
    }

    toast_text = ""
    toast_time = 0

    # fPS tracking
    fps_times = []

    # Two-hand exit 
    two_hand_count = 0

    print("\n==============================================")
    print("  AIR CANVAS - Gap-Free Edition")
    print("==============================================")
    print("  Index finger ONLY   ->  DRAW")
    print("  Index + Middle UP   ->  SELECT / TOOLBAR")
    print("  Everything else     ->  PEN LIFT")
    print("  Both hands (~0.5s)  ->  Return to menu")
    print("  Press 'q' or ESC to exit")
    print("  Press 'c' to clear canvas")
    print("  Press 'z' to undo  or  'y' to redo")
    print("  Press '+'/'-' to adjust brush size")
    print("==============================================\n")

    vdoptr = cv.VideoCapture(0)
    if not vdoptr.isOpened():
        print("[ERROR] Cannot open camera for Air Canvas.")
        return

    vdoptr.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    vdoptr.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # fullscreen window
    cv.namedWindow("AirCanvas", cv.WINDOW_NORMAL)
    cv.setWindowProperty("AirCanvas", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    detector = htm.handDetector(detectionCon=0.85)

    while True:
        success, frame = vdoptr.read()
        if not success:
            continue

        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape

        if imgCanvas is None:
            imgCanvas = np.zeros_like(frame)

        if 'buttons' not in state:
            num_btns = 6
            margin = 12
            bw = (w - (num_btns + 1) * margin) // num_btns
            bh = int(h * 0.10)
            by1, by2 = 12, 12 + bh
            state['buttons'] = [
                {"label": "CLEAR",   "color": (80,  80,  80)},
                {"label": "RED",     "color": (0,   0,   255)},
                {"label": "GREEN",   "color": (0,   255, 0)},
                {"label": "BLUE",    "color": (255, 0,   0)},
                {"label": "RAINBOW", "color": (255, 0,   255)},
                {"label": "ERASER",  "color": (20,  20,  20)},
            ]
            for i, b in enumerate(state['buttons']):
                b['y1'] = by1
                b['y2'] = by2
                b['x1'] = margin + i * (bw + margin)
                b['x2'] = b['x1'] + bw
                b['real_color'] = b['color']

        #keyboard commands 
        key = cv.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('c'):
            canvas_history.append(imgCanvas.copy())
            if len(canvas_history) > MAX_UNDO:
                canvas_history.pop(0)
            imgCanvas = np.zeros_like(frame)
            current_stroke = None
            redo_stack.clear()
            toast_text = "Canvas Cleared"
            toast_time = time.time()
        elif key == ord('z'):
            if canvas_history:
                redo_stack.append(imgCanvas.copy())
                imgCanvas = canvas_history.pop()
                current_stroke = None
                toast_text = f"Undo ({len(canvas_history)} left)"
                toast_time = time.time()
        elif key == ord('y'):
            if redo_stack:
                canvas_history.append(imgCanvas.copy())
                imgCanvas = redo_stack.pop()
                current_stroke = None
                toast_text = "Redo"
                toast_time = time.time()
        elif key in [ord('+'), ord('=')]:
            state["brushThickness"] = min(40, state["brushThickness"] + 2)
            toast_text = f"Brush: {state['brushThickness']}px"
            toast_time = time.time()
        elif key in [ord('-'), ord('_')]:
            state["brushThickness"] = max(2, state["brushThickness"] - 2)
            toast_text = f"Brush: {state['brushThickness']}px"
            toast_time = time.time()

        # Hand detection
        frame = detector.findHands(frame, draw=False)

        # Two-hand exit
        if (detector.results.multi_hand_landmarks and len(detector.results.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — returning to menu.")
                break
        else:
            two_hand_count = 0

        lmlist = detector.findPosition(frame, draw=False)

        if len(lmlist) != 0:
            fingers = detector.fingersUp()
            index_up = fingers[1] == 1
            middle_up = fingers[2] == 1

            ix, iy = lmlist[8][1], lmlist[8][2]
            mx2, my2 = lmlist[12][1], lmlist[12][2]

            if index_up and middle_up:
                raw_gesture = "SELECT"
            elif index_up and not middle_up:
                raw_gesture = "DRAW"
            else:
                raw_gesture = "LIFT"

            if raw_gesture == pending_mode:
                buffer_count += 1
            else:
                pending_mode = raw_gesture
                buffer_count = 1

            if pending_mode == "DRAW":
                buf_required = BUFFER_ENTER_DRAW
            elif pending_mode == "SELECT":
                buf_required = BUFFER_ENTER_SEL
            else:
                buf_required = 2

            if state["mode"] == "DRAW" and raw_gesture != "DRAW":
                new_mode = "LIFT"
                if current_stroke is not None:
                    if len(current_stroke["points"]) > 0:
                        canvas_history.append(imgCanvas.copy())
                        if len(canvas_history) > MAX_UNDO:
                            canvas_history.pop(0)
                        bake_stroke_to_canvas(imgCanvas, current_stroke)
                        redo_stack.clear()
                    current_stroke = None
                lift_cooldown = POST_LIFT_FRAMES
            elif buffer_count >= buf_required:
                new_mode = pending_mode
            else:
                new_mode = state["mode"]

            if lift_cooldown > 0:
                lift_cooldown -= 1
                if new_mode == "DRAW":
                    new_mode = "LIFT"

            committed_mode = new_mode

            if committed_mode == "SELECT":
                state["mode"] = "SELECT"
                for b in state['buttons']:
                    if b['x1'] < ix < b['x2'] and b['y1'] < iy < b['y2']:
                        if b['label'] == "CLEAR":
                            canvas_history.append(imgCanvas.copy())
                            if len(canvas_history) > MAX_UNDO:
                                canvas_history.pop(0)
                            imgCanvas = np.zeros_like(frame)
                            current_stroke = None
                            redo_stack.clear()
                        elif b['label'] == "ERASER":
                            state["drawcolor"] = (0, 0, 0)
                            state["rainbow"] = False
                        elif b['label'] == "RAINBOW":
                            state["rainbow"] = True
                        else:
                            state["drawcolor"] = b['real_color']
                            state["rainbow"] = False

                sel_c = get_rainbow_color() if state["rainbow"] else state["drawcolor"]
                if sel_c == (0, 0, 0):
                    sel_c = (200, 200, 200)
                cv.circle(frame, (ix, iy), 10, sel_c, cv.FILLED)

            elif committed_mode == "DRAW":
                state["mode"] = "DRAW"
                d_color = get_rainbow_color() if state["rainbow"] else state["drawcolor"]
                d_thick = eraserThickness if d_color == (0, 0, 0) else state["brushThickness"]

                draw_x, draw_y = ix, iy

                if current_stroke is None:
                    current_stroke = {
                        "color": d_color,
                        "thickness": d_thick,
                        "points": deque(maxlen=4096),
                    }
                current_stroke["points"].append((draw_x, draw_y))
                if state["rainbow"]:
                    current_stroke["color"] = d_color

                if d_color == (0, 0, 0):
                    cv.circle(frame, (draw_x, draw_y), eraserThickness // 2, (200, 200, 200), 2)
                else:
                    cv.circle(frame, (draw_x, draw_y), max(d_thick // 2, 4), d_color, cv.FILLED)

            else:
                if state["mode"] == "DRAW" and current_stroke is not None:
                    if len(current_stroke["points"]) > 0:
                        canvas_history.append(imgCanvas.copy())
                        if len(canvas_history) > MAX_UNDO:
                            canvas_history.pop(0)
                        bake_stroke_to_canvas(imgCanvas, current_stroke)
                        redo_stack.clear()
                    current_stroke = None
                state["mode"] = "LIFT"

        else:
            if current_stroke is not None:
                if len(current_stroke["points"]) > 0:
                    canvas_history.append(imgCanvas.copy())
                    if len(canvas_history) > MAX_UNDO:
                        canvas_history.pop(0)
                    bake_stroke_to_canvas(imgCanvas, current_stroke)
                    redo_stack.clear()
                current_stroke = None
            state["mode"] = "LIFT"
            pending_mode = "LIFT"
            buffer_count = 0
            lift_cooldown = 0

        #Toolbar draw 
        if 'buttons' in state:
            overlay = frame.copy()
            for b in state['buttons']:
                is_active = (
                    (state["rainbow"] and b["label"] == "RAINBOW") or
                    (not state["rainbow"] and state["drawcolor"] == (0, 0, 0) and b["label"] == "ERASER") or
                    (not state["rainbow"] and state["drawcolor"] == b['real_color'] and b["label"] not in ["CLEAR", "ERASER", "RAINBOW"])
                )
                btn_c = get_rainbow_color() if b["label"] == "RAINBOW" else b["color"]
                cv.rectangle(overlay, (b["x1"], b["y1"]), (b["x2"], b["y2"]), btn_c, cv.FILLED)
                if is_active:
                    cv.rectangle(overlay, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255, 255, 255), 2)
                cv.putText(overlay, b["label"], (b["x1"] + 10, b["y1"] + 35), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frame = cv.addWeighted(overlay, 0.85, frame, 0.15, 0)

        final = composite_canvas(frame, imgCanvas, current_stroke)

        # fps and user interface
        now = time.time()
        fps_times.append(now)
        fps_times[:] = [t for t in fps_times if now - t < 1.0]
        cv.putText(final, f"FPS: {len(fps_times)}", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if toast_text and time.time() - toast_time < 1.5:
            cv.putText(final, toast_text, (w // 2 - 50, h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        undo_count = len(canvas_history)
        bar = f"Brush: {state['brushThickness']}px  |  Undo: {undo_count}  |  z=undo y=redo c=clear q=quit"
        cv.putText(final, bar, (15, h - 15), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv.imshow("AirCanvas", final)

    vdoptr.release()
    cv.destroyAllWindows()
    print("Air Canvas closed.")


if __name__ == "__main__":
    run_canvas()