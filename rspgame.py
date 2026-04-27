"""
  - SPACE = start round
  - 'r'   = reset scores
  - 'q'   = quit / back to menu
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import random
import time
import math

mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

CANVAS_W = 900
CANVAS_H = 620

# 
BG_COLOR      = (18, 18, 22)
ACCENT        = (0, 220, 255)      # cyan
WIN_COLOR     = (0, 220, 100)
LOSE_COLOR    = (0, 80, 220)
DRAW_COLOR    = (180, 180, 100)
TEXT_DIM      = (120, 120, 130)
TEXT_BRIGHT   = (230, 230, 240)


COUNTDOWN_SECS = 3
RESULT_DISPLAY = 2.5   # seconds to show result before allowing next round

GESTURE_NAMES = ["ROCK", "PAPER", "SCISSORS"]



def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    overlay = img.copy()
    cv.rectangle(overlay, (x + r, y),     (x + w - r, y + h),     color, -1)
    cv.rectangle(overlay, (x, y + r),     (x + w,     y + h - r), color, -1)
    cv.circle(overlay, (x + r,     y + r),     r, color, -1)
    cv.circle(overlay, (x + w - r, y + r),     r, color, -1)
    cv.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    if alpha < 1.0:
        cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        np.copyto(img, overlay)


def detect_gesture(hand_landmarks):
    """
    Detect Rock, Paper, or Scissors from hand landmarks.

    """
    lm = hand_landmarks.landmark

   

    fingers_up = []

    thumb_tip = lm[4]
    thumb_ip  = lm[3]
    thumb_mcp = lm[2]
    fingers_up.append(abs(thumb_tip.x - thumb_mcp.x) > 0.05)

    for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_up.append(lm[tip_id].y < lm[pip_id].y)

    thumb, index, middle, ring, pinky = fingers_up

    total_up = sum([index, middle, ring, pinky]) 

    if total_up == 0:
        return "rock"

    if index and middle and not ring and not pinky:
        return "scissors"

    if total_up >= 3:
        return "paper"

    return None  


def get_winner(player, computer):
    if player == computer:
        return "draw"
    wins = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if wins[player] == computer:
        return "player"
    return "computer"


def draw_gesture_icon(canvas, gesture, cx, cy, size, color):
    """Draw a simple visual representation of the gesture."""
    if gesture == "rock":
        cv.circle(canvas, (cx, cy), size, color, -1)
        cv.circle(canvas, (cx, cy), size, (255, 255, 255), 2)
        cv.putText(canvas, "ROCK", (cx - 30, cy + size + 25),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    elif gesture == "paper":
        cv.rectangle(canvas, (cx - size, cy - size),(cx + size, cy + size), color, -1)
        cv.rectangle(canvas, (cx - size, cy - size),(cx + size, cy + size), (255, 255, 255), 2)
        cv.putText(canvas, "PAPER", (cx - 35, cy + size + 25),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    elif gesture == "scissors":
        cv.line(canvas, (cx - size, cy + size), (cx + size//2, cy - size), color, 4)
        cv.line(canvas, (cx + size, cy + size), (cx - size//2, cy - size), color, 4)
        cv.circle(canvas, (cx, cy), 6, color, -1)
        cv.putText(canvas, "SCISSORS", (cx - 50, cy + size + 25),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ─── MAIN ─────────────────────────────────────────────────────

def run_rps():
  
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for Rock Paper Scissors.")
        return
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.namedWindow("Rock Paper Scissors", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Rock Paper Scissors", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    player_score   = 0
    computer_score = 0
    draws          = 0
    rounds_played  = 0

    STATE_IDLE      = 0
    STATE_COUNTDOWN = 1
    STATE_RESULT    = 2

    state           = STATE_IDLE
    countdown_start = 0
    result_start    = 0

    player_gesture  = None
    computer_choice = None
    round_result    = None     
    current_gesture = None     

    print("\n============================================")
    print("  ROCK PAPER SCISSORS")
    print("============================================")
    print("  Press SPACE to start a round!")
    print("  Show: Fist=Rock, Open=Paper, Peace=Scissors")
    print("  'r' = reset  |  'q' = quit")
    print("============================================\n")

    two_hand_count = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands_model.process(rgb)

        # Two-hand exit: show both hands for 0.5 s
        if (result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — returning to menu.")
                break
        else:
            two_hand_count = 0

        now = time.time()

        current_gesture = None
        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            current_gesture = detect_gesture(hand_lm)

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

        # Title
        cv.putText(canvas, "ROCK  PAPER  SCISSORS",(CANVAS_W // 2 - 210, 45),cv.FONT_HERSHEY_SIMPLEX, 1.0, ACCENT, 2, cv.LINE_AA)

        cam_w, cam_h = 280, 210
        cam_x = (CANVAS_W - cam_w) // 2
        cam_y = 65
        cam_small = cv.resize(frame, (cam_w, cam_h))

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                temp = frame.copy()
                mp_draw.draw_landmarks(temp, hand_lm, mp_hands.HAND_CONNECTIONS)
                cam_small = cv.resize(temp, (cam_w, cam_h))

        canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = cam_small
        cv.rectangle(canvas, (cam_x - 2, cam_y - 2),(cam_x + cam_w + 2, cam_y + cam_h + 2), (60, 60, 70), 2)

        if current_gesture and state != STATE_RESULT:
            gesture_label = current_gesture.upper()
            (gw, _), _ = cv.getTextSize(gesture_label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv.putText(canvas, f"Detected: {gesture_label}", (CANVAS_W // 2 - 80, cam_y + cam_h + 28),cv.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT, 1, cv.LINE_AA)
        elif state != STATE_RESULT:
            cv.putText(canvas, "Show your hand!",(CANVAS_W // 2 - 85, cam_y + cam_h + 28),cv.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 1)

        score_y = 310
        draw_rounded_rect(canvas, 30, score_y, CANVAS_W - 60, 70, 10, (30, 30, 38))

        cv.putText(canvas, f"YOU: {player_score}", (60, score_y + 45),cv.FONT_HERSHEY_SIMPLEX, 0.8, WIN_COLOR, 2)

        cv.putText(canvas, f"DRAW: {draws}", (CANVAS_W // 2 - 55, score_y + 45),cv.FONT_HERSHEY_SIMPLEX, 0.7, DRAW_COLOR, 1)

        cv.putText(canvas, f"CPU: {computer_score}", (CANVAS_W - 220, score_y + 45),cv.FONT_HERSHEY_SIMPLEX, 0.8, LOSE_COLOR, 2)

        cv.putText(canvas, f"Rounds: {rounds_played}",(CANVAS_W // 2 - 50, score_y + 68),cv.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1)

        
        if state == STATE_IDLE:
            prompt = "Press SPACE to play!"
            (pw, _), _ = cv.getTextSize(prompt, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            pulse = abs(math.sin(now * 2.5))
            alpha_val = int(150 + 105 * pulse)
            p_color = (0, alpha_val, 255)
            cv.putText(canvas, prompt, ((CANVAS_W - pw) // 2, 430),cv.FONT_HERSHEY_SIMPLEX, 0.9, p_color, 2, cv.LINE_AA)

        elif state == STATE_COUNTDOWN:
            elapsed = now - countdown_start
            remaining = COUNTDOWN_SECS - elapsed

            if remaining > 0:
                count_num = int(remaining) + 1
                text = str(count_num)
                (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 4, 6)
                tx = (CANVAS_W - tw) // 2
                ty = 480
                cv.putText(canvas, text, (tx + 3, ty + 3),cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 6)
                cv.putText(canvas, text, (tx, ty),cv.FONT_HERSHEY_SIMPLEX, 4, ACCENT, 6)

                cv.putText(canvas, "Get ready — show your gesture!",(CANVAS_W // 2 - 185, 520),cv.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_DIM, 1)

                bar_w = int((CANVAS_W - 100) * (1.0 - remaining / COUNTDOWN_SECS))
                cv.rectangle(canvas, (50, 550), (50 + bar_w, 560), ACCENT, -1)
                cv.rectangle(canvas, (50, 550), (CANVAS_W - 50, 560), (60, 60, 70), 1)
            else:
                player_gesture = current_gesture if current_gesture else "rock"
                computer_choice = random.choice(["rock", "paper", "scissors"])
                round_result = get_winner(player_gesture, computer_choice)

                if round_result == "player":
                    player_score += 1
                elif round_result == "computer":
                    computer_score += 1
                else:
                    draws += 1
                rounds_played += 1

                state = STATE_RESULT
                result_start = now

        elif state == STATE_RESULT:
            elapsed = now - result_start

            draw_gesture_icon(canvas, player_gesture, 180, 470, 40, (0, 200, 255))
            cv.putText(canvas, "YOU", (155, 415),cv.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_BRIGHT, 2)

            cv.putText(canvas, "VS", (CANVAS_W // 2 - 20, 475),cv.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_DIM, 2)

            draw_gesture_icon(canvas, computer_choice, CANVAS_W - 180, 470, 40, (200, 100, 255))
            cv.putText(canvas, "CPU", (CANVAS_W - 205, 415),cv.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_BRIGHT, 2)

            if round_result == "player":
                result_text = "YOU WIN!"
                result_color = WIN_COLOR
            elif round_result == "computer":
                result_text = "CPU WINS!"
                result_color = LOSE_COLOR
            else:
                result_text = "DRAW!"
                result_color = DRAW_COLOR

            (rw, _), _ = cv.getTextSize(result_text, cv.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv.putText(canvas, result_text,((CANVAS_W - rw) // 2, 580),cv.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 3, cv.LINE_AA)

            if elapsed > RESULT_DISPLAY:
                state = STATE_IDLE

        cv.putText(canvas, "SPACE=play  R=reset  Q=quit",(CANVAS_W // 2 - 160, CANVAS_H - 12),cv.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 80), 1)

        cv.imshow("Rock Paper Scissors", canvas)

        key = cv.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord(' ') and state == STATE_IDLE:
            state = STATE_COUNTDOWN
            countdown_start = time.time()
            player_gesture = None
            computer_choice = None
            round_result = None
        elif key == ord('r'):
            player_score = 0
            computer_score = 0
            draws = 0
            rounds_played = 0
            state = STATE_IDLE

    cap.release()
    cv.destroyWindow("Rock Paper Scissors")
    print("Rock Paper Scissors closed. Returning to menu...")


if __name__ == "__main__":
    run_rps()
