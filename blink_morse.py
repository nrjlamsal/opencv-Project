"""
👁️ Blink Morse Code
Uses webcam + face mesh to detect blinks and decode Morse code.

How it works:
  - Short blink (< 0.4s) is  DOT  (·)
  - Long blink  (0.4 to 1.5s) is DASH (-)
  - Eyes open for 1.5s decodes current morse to letter
  - Eyes open for 3.0s adds space between words
  - Press 'c' to clear, 'q' to quit


"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

#  (used only for two hand exit gesture)
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# configuration of window
CANVAS_W = 1000
CANVAS_H = 650

# Eye Aspect Ratio threshold
EAR_THRESHOLD = 0.21       # below this = eyes closed = blink

# Timing thresholds in seconds
SHORT_BLINK_MAX  = 0.4     # blink shorter than this = DOT
LONG_BLINK_MIN   = 0.4     # blink longer than this = DASH
LONG_BLINK_MAX   = 1.5     # blinks longer than this are ignored (natural long close)
LETTER_PAUSE     = 1.5     # eyes open this long = decode current morse sequence
WORD_PAUSE       = 3.0     # eyes open this long = add space

# Colors
BG_COLOR    = (18, 18, 22)
ACCENT      = (0, 220, 255)
DOT_COLOR   = (0, 200, 255)
DASH_COLOR  = (255, 180, 0)
TEXT_BRIGHT  = (230, 230, 240)
TEXT_DIM     = (120, 120, 130)
SUCCESS      = (0, 220, 100)
ERROR_COLOR  = (0, 80, 220)

# Eye landmark indices for mediapipe
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ─── MORSE CODE TABLE ────────────────────────────────────────
MORSE_TO_CHAR = {
    ".-":    "A",  "-...":  "B",  "-.-.":  "C",  "-..":   "D",
    ".":     "E",  "..-.":  "F",  "--.":   "G",  "....":  "H",
    "..":    "I",  ".---":  "J",  "-.-":   "K",  ".-..":  "L",
    "--":    "M",  "-.":    "N",  "---":   "O",  ".--.":  "P",
    "--.-":  "Q",  ".-.":   "R",  "...":   "S",  "-":     "T",
    "..-":   "U",  "...-":  "V",  ".--":   "W",  "-..-":  "X",
    "-.--":  "Y",  "--..":  "Z",
    "-----": "0",  ".----": "1",  "..---": "2",  "...--": "3",
    "....-": "4",  ".....": "5",  "-....": "6",  "--...": "7",
    "---..": "8",  "----.": "9",
}

CHAR_TO_MORSE = {v: k for k, v in MORSE_TO_CHAR.items()}



def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    calculate Eye Aspect Ratio (EAR).
    Low EAR = eye closed, High EAR = eye open.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))

    # Vertical distances
    v1 = math.dist(pts[1], pts[5])  
    v2 = math.dist(pts[2], pts[4]) 
    # Horizontal distance
    h1 = math.dist(pts[0], pts[3])  

    if h1 == 0:
        return 0.3 

    ear = (v1 + v2) / (2.0 * h1)
    return ear


def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    """Draw a filled rectangle with rounded corners"""
    overlay = img.copy()
    cv.rectangle(overlay, (x + r, y),(x + w - r, y + h), color, -1)
    cv.rectangle(overlay, (x, y + r),(x + w,y + h - r),color, -1)
    cv.circle(overlay, (x + r,     y + r),r, color, -1)
    cv.circle(overlay, (x + w - r, y + r),r, color, -1)
    cv.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv.circle(overlay, (x + w - r, y + h - r), r, color, -1)

    if alpha < 1.0:
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        np.copyto(img, overlay)


def morse_to_visual(morse_seq):
    """Convert morse string  to spaced dots and dashes for display"""
    visual = ""
    for ch in morse_seq:
        if ch == ".":
            visual += ".  "
        elif ch == "-":
            visual += "--  "
    return visual.strip()

def run_morse():
    """
    Launch the Blink Morse Code communication tool.
    """
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for Blink Morse Code.")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.namedWindow("Blink Morse Code", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Blink Morse Code", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # State
    morse_active     = False    
    eyes_closed      = False
    blink_start      = 0        
    eyes_open_since  = time.time()  

    current_morse    = ""       
    decoded_text     = ""      
    last_decoded     = ""       
    last_decode_time = 0

    # History of dots and dashes for visual display
    morse_history    = []       
    letter_added     = False    
    space_added      = False

    ear_buffer = []
    EAR_BUFFER_SIZE = 3

    print("\n============================================")
    print("  BLINK MORSE CODE")
    print("============================================")
    print("  SPACE = start/pause ")
    print("  Short blink = DOT (.)")
    print("  Long blink  = DASH (-)")
    print("  Pause 1.5s  = decode letter")
    print("  Pause 3.0s  = space")
    print("  'c' = clear  |  'q' = quit")
    print("============================================\n")

    two_hand_count = 0  #  counter for two-hand exit gesture

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Two-hand exit
        hand_result = hands_model.process(rgb)
        if (hand_result.multi_hand_landmarks
                and len(hand_result.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — returning to menu.")
                break
        else:
            two_hand_count = 0

        now = time.time()

        # building canvas
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

        # Title
        cv.putText(canvas, "BLINK MORSE CODE",(CANVAS_W // 2 - 180, 42),cv.FONT_HERSHEY_SIMPLEX, 1.0, ACCENT, 2, cv.LINE_AA)

        #Camera feed 
        cam_w, cam_h = 260, 195
        cam_x = CANVAS_W - cam_w - 25
        cam_y = 65

        ear_value = 0.3  #
        face_detected = False

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]
            face_detected = True

            # calculate EAR for both eyes
            left_ear  = eye_aspect_ratio(face_lm.landmark, LEFT_EYE, fw, fh)
            right_ear = eye_aspect_ratio(face_lm.landmark, RIGHT_EYE, fw, fh)
            ear_value = (left_ear + right_ear) / 2.0

            ear_buffer.append(ear_value)
            if len(ear_buffer) > EAR_BUFFER_SIZE:
                ear_buffer.pop(0)
            ear_smooth = sum(ear_buffer) / len(ear_buffer)

            # draw eye landmarks on frame
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = face_lm.landmark[idx]
                px = int(lm.x * fw)
                py = int(lm.y * fh)
                color = (0, 0, 255) if ear_smooth < EAR_THRESHOLD else (0, 255, 0)
                cv.circle(frame, (px, py), 2, color, -1)

            currently_closed = ear_smooth < EAR_THRESHOLD

            if morse_active:
                if currently_closed and not eyes_closed:
                    eyes_closed = True
                    blink_start = now
                    letter_added = False
                    space_added = False

                elif not currently_closed and eyes_closed:
                    eyes_closed = False
                    blink_duration = now - blink_start
                    eyes_open_since = now

                    if blink_duration < SHORT_BLINK_MAX:
                        # DOT
                        current_morse += "."
                        morse_history.append((".", now))
                    elif blink_duration < LONG_BLINK_MAX:
                        # DASH
                        current_morse += "-"
                        morse_history.append(("-", now))

                elif not currently_closed and not eyes_closed:
                    open_duration = now - eyes_open_since

                    if current_morse and open_duration >= LETTER_PAUSE and not letter_added:
                        decoded_char = MORSE_TO_CHAR.get(current_morse, "?")
                        decoded_text += decoded_char
                        last_decoded = decoded_char
                        last_decode_time = now
                        morse_history.append((f">{decoded_char}", now))
                        current_morse = ""
                        letter_added = True
                        space_added = False

                    if open_duration >= WORD_PAUSE and not space_added and letter_added:
                        decoded_text += " "
                        morse_history.append(("_", now))
                        space_added = True
            else:
                eyes_closed = currently_closed
                eyes_open_since = now

        # Camera preview
        cam_small = cv.resize(frame, (cam_w, cam_h))
        canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = cam_small
        cv.rectangle(canvas, (cam_x - 2, cam_y - 2),(cam_x + cam_w + 2, cam_y + cam_h + 2), (60, 60, 70), 2)

        # Eye status indicator
        if face_detected:
            status = "CLOSED" if eyes_closed else "OPEN"
            s_color = ERROR_COLOR if eyes_closed else SUCCESS
            cv.putText(canvas, f"Eyes: {status}", (cam_x, cam_y + cam_h + 22),cv.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)

            bar_x = cam_x
            bar_y = cam_y + cam_h + 32
            bar_w = cam_w
            bar_h = 12
            cv.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 50), -1)
            fill_w = int(bar_w * min(ear_value / 0.35, 1.0))
            bar_color = (0, 200, 100) if ear_value >= EAR_THRESHOLD else (0, 80, 220)
            cv.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            cv.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 80), 1)
            # threshold line
            thresh_x = bar_x + int(bar_w * (EAR_THRESHOLD / 0.35))
            cv.line(canvas, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 1)
        else:
            cv.putText(canvas, "No face detected", (cam_x, cam_y + cam_h + 22),cv.FONT_HERSHEY_SIMPLEX, 0.5, ERROR_COLOR, 1)

        # paused indicator
        if morse_active:
            ind_color = (0, 200, 100)
            ind_text = "ACTIVE - Blinks recording"
            # Green glowing border
            cv.rectangle(canvas, (2, 2), (CANVAS_W - 2, CANVAS_H - 2), ind_color, 2)
        else:
            ind_color = (0, 80, 200)
            ind_text = "PAUSED - Press SPACE to start"

        badge_w = 320
        draw_rounded_rect(canvas, 40, 55, badge_w, 30, 6, ind_color, alpha=0.85)
        cv.putText(canvas, ind_text, (50, 76),cv.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv.LINE_AA)

        # current morse sequence 
        morse_panel_x = 40
        morse_panel_y = 75
        draw_rounded_rect(canvas, morse_panel_x, morse_panel_y,CANVAS_W - cam_w - 80, 110, 10, (28, 28, 35))

        cv.putText(canvas, "Current Morse:", (morse_panel_x + 15, morse_panel_y + 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1)

        if current_morse:
            # Show raw morse code clearly
            spaced = "  ".join(list(current_morse))
            cv.putText(canvas, spaced, (morse_panel_x + 15, morse_panel_y + 55), cv.FONT_HERSHEY_SIMPLEX, 1.5, ACCENT, 3, cv.LINE_AA)

            # Show visual: dots as filled circles, dashes as lines
            vx = morse_panel_x + 15
            vy = morse_panel_y + 90
            for ch in current_morse:
                if ch == ".":
                    cv.circle(canvas, (vx + 8, vy), 8, DOT_COLOR, -1)
                    vx += 30
                elif ch == "-":
                    cv.line(canvas, (vx, vy), (vx + 35, vy), DASH_COLOR, 6)
                    vx += 50
        else:
            cv.putText(canvas, "Blink to start...", (morse_panel_x + 15, morse_panel_y + 65),cv.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 70), 1)

        #  Blink timer
        if eyes_closed:
            blink_dur = now - blink_start
            indicator_y = 205
            cv.putText(canvas, f"Blinking: {blink_dur:.1f}s", (morse_panel_x, indicator_y),cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Zone indicator
            if blink_dur < SHORT_BLINK_MAX:
                cv.putText(canvas, "-> DOT", (morse_panel_x + 200, indicator_y),cv.FONT_HERSHEY_SIMPLEX, 0.6, DOT_COLOR, 2)
            elif blink_dur < LONG_BLINK_MAX:
                cv.putText(canvas, "-> DASH", (morse_panel_x + 200, indicator_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, DASH_COLOR, 2)
            else:
                cv.putText(canvas, "-> (too long, ignored)", (morse_panel_x + 200, indicator_y),cv.FONT_HERSHEY_SIMPLEX, 0.5, ERROR_COLOR, 1)

        # decoded text display
        text_y = 250
        draw_rounded_rect(canvas, 30, text_y, CANVAS_W - 60, 80, 10, (25, 25, 32))
        cv.rectangle(canvas, (30, text_y), (CANVAS_W - 30, text_y + 80), (60, 60, 70), 1)

        cv.putText(canvas, "Decoded Text:", (50, text_y + 22),cv.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1)

        display_text = decoded_text if decoded_text else "(blink to spell...)"
        text_color = TEXT_BRIGHT if decoded_text else (50, 50, 60)
        # Truncate
        if len(display_text) > 50:
            display_text = "..." + display_text[-47:]
        cv.putText(canvas, display_text, (50, text_y + 58),cv.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 1, cv.LINE_AA)

        # Last decoded character
        if last_decoded and now - last_decode_time < 1.5:
            pulse = abs(math.sin((now - last_decode_time) * 5))
            char_color = (0, int(200 + 55 * pulse), int(100 + 155 * pulse))
            cv.putText(canvas, last_decoded, (CANVAS_W - 120, text_y + 65),cv.FONT_HERSHEY_SIMPLEX, 2.0, char_color, 3)

        # Morse reference table
        ref_y = 355
        cv.putText(canvas, "Morse Code Reference:", (40, ref_y),cv.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1)

        ref_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        cols = 7
        for i, ch in enumerate(ref_chars):
            col = i % cols
            row = i // cols
            rx = 40 + col * 135
            ry = ref_y + 30 + row * 30

            morse = CHAR_TO_MORSE.get(ch, "")
            spaced_morse = " ".join(list(morse))
            cv.putText(canvas, f"{ch}:", (rx, ry),cv.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_BRIGHT, 1)

            # Draw dots as small circles, dashes as short lines
            sx = rx + 30
            sy = ry - 5
            for symbol in morse:
                if symbol == ".":
                    cv.circle(canvas, (sx + 4, sy), 4, DOT_COLOR, -1)
                    sx += 16
                elif symbol == "-":
                    cv.line(canvas, (sx, sy), (sx + 16, sy), DASH_COLOR, 3)
                    sx += 24

        log_y = ref_y + 140
        cv.putText(canvas, "Activity:", (40, log_y),cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1)

        # shows last 15 events
        recent = [sym for sym, t in morse_history if now - t < 30][-15:]
        if recent:
            activity_str = "  ".join(recent)
            cv.putText(canvas, activity_str, (130, log_y),cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_BRIGHT, 1)

        #  Instructions
        instructions = [
            "SPACE = start/stop input",
            "Short blink (<0.4s) = DOT",
            "Long blink (0.4-1.5s) = DASH",
            "Pause 1.5s = decode letter",
            "Pause 3.0s = add space",
            "'c' = clear  |  'q' = quit",
        ]
        for i, line in enumerate(instructions):
            cv.putText(canvas, line, (CANVAS_W - cam_w - 20, CANVAS_H - 120 + i * 22),cv.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 80), 1)
        cv.imshow("Blink Morse Code", canvas)

        # Keyboard 
        key = cv.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord(' '):
            morse_active = not morse_active
            if morse_active:
                eyes_open_since = time.time()
                eyes_closed = False
                current_morse = ""
                letter_added = False
                space_added = False
                print("  >> MORSE INPUT ACTIVE")
            else:
                print("  >> MORSE INPUT PAUSED")
        elif key == ord('c'):
            decoded_text = ""
            current_morse = ""
            morse_history.clear()
            last_decoded = ""

    cap.release()
    cv.destroyWindow("Blink Morse Code")
    print(f"Blink Morse Code closed. You typed: '{decoded_text}'")


if __name__ == "__main__":
    run_morse()
