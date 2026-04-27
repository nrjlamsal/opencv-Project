"""
✋ Air Keyboard — Hand-Controlled Virtual Keyboard
  - Move index finger       → hover / aim at keys
  - Hold on a key (~1s)     → key presses (progress ring shows countdown)
  - Hold on a suggestion    → auto-completes the word
  - Move away               → cancels the press
  - Show open palm          → toggle keyboard layout (letters ↔ numbers/symbols)
  - Press ESC               → quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

LAYOUT_LETTERS = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",";"],
    ["Z","X","C","V","B","N","M",",",".","Backspace"],
    ["SPACE","ENTER"],
]

LAYOUT_NUMBERS = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["!","@","#","$","%","^","&","*","(", ")"],
    ["-","_","=","+","[","]","{","}","\\","/"],
    ["SPACE","ENTER"],
]

CAM_W, CAM_H       = 1280, 720
KEY_W, KEY_H       = 90, 70         
KEY_MARGIN         = 8
KB_ORIGIN_X        = 40           
KB_ORIGIN_Y        = 305            
HOVER_COLOR        = (255, 200, 50)
PRESS_COLOR        = (50, 220, 120)
KEY_BG_COLOR       = (30, 30, 50)
KEY_TEXT_COLOR      = (230, 230, 255)
PALM_OPEN_THRESH   = 5             

SUGGESTION_Y       = 260            
SUGGESTION_H       = 36             
SUGGESTION_GAP     = 10
SUGGESTION_COLOR   = (45, 35, 55)
SUGGESTION_HOVER   = (80, 60, 100)
SUGGESTION_ACCENT  = (200, 150, 255)
NUM_SUGGESTIONS    = 3

# Dwell timer 
DWELL_TIME         = 1            # seconds to hold on a key 
DWELL_COOLDOWN     = 1            # seconds after a press before same key can be pressed

pyautogui.FAILSAFE = False


WORD_LIST = sorted(set([
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her","she",
    "or","an","will","my","one","all","would","there","their","what","so","up","out","if",
    "about","who","get","which","go","me","when","make","can","like","time","no","just",
    "him","know","take","people","into","year","your","good","some","could","them","see",
    "other","than","then","now","look","only","come","its","over","think","also","back",
    "after","use","two","how","our","work","first","well","way","even","new","want",
    "because","any","these","give","day","most","us","great","between","need","large",
    "very","find","here","thing","many","right","still","much","hand","high","keep",
    "last","long","made","world","before","should","through","life","where","after",
    "help","home","never","best","old","off","been","put","must","big","end","while",
    "turn","real","leave","might","open","begin","run","show","every","small","number",
    "again","point","found","study","name","play","change","move","try","close","door",
    "line","part","school","start","city","head","side","water","group","body","music",
    "hard","game","face","book","word","write","read","place","story","fact","love",
    "stop","once","late","since","call","idea","ask","kind","food","house","own","left",
    "team","room","may","power","car","money","area","state","family","country","student",
    "problem","system","program","question","during","company","develop","important",
    "market","service","member","meeting","report","project","business","computer",
    "information","technology","research","experience","education","environment",
    "thank","thanks","hello","please","sorry","welcome","today","tomorrow","yesterday",
    "morning","night","week","month","happy","friend","together","always","already",
    "nothing","something","everything","everyone","someone","anyone","anything",
    "beautiful","wonderful","different","possible","available","certain","simple",
    "special","working","looking","making","going","coming","getting","having","being",
    "using","taking","doing","saying","playing","running","trying","thinking","learning",
    "building","creating","writing","reading","walking","talking","sitting","standing",
    "should","could","would","might","really","actually","probably","definitely",
    "university","keyboard","canvas","python","project","gesture","camera","finger",
    "screen","button","window","color","image","video","sound","data","code","file",
    "class","function","method","object","value","type","list","array","string",
    "number","letter","message","email","phone","website","internet","online",
    "awesome","amazing","perfect","cool","nice","sure","okay","yes","no","maybe",
]))


def get_suggestions(typed_text, n=NUM_SUGGESTIONS):
    """
    Get word suggestions based on the current partial word being typed.

    """
    # Extract the last partial word
    words = typed_text.lower().split()
    if not words or typed_text.endswith(" "):
        return []

    partial = words[-1]
    if len(partial) < 1:
        return []

    
    matches = [w for w in WORD_LIST if w.startswith(partial) and w != partial]

    matches.sort(key=lambda w: (len(w), w))

    return matches[:n]


def build_suggestion_rects(suggestions):
    """
    Build clickable rectangles for the suggestion bar above the keyboard.

    """
    if not suggestions:
        return []

    total_keys_w = len(LAYOUT_LETTERS[0]) * (KEY_W + KEY_MARGIN) - KEY_MARGIN
    btn_w = (total_keys_w - (len(suggestions) - 1) * SUGGESTION_GAP) // len(suggestions)
    rects = []
    for i, word in enumerate(suggestions):
        x1 = KB_ORIGIN_X + i * (btn_w + SUGGESTION_GAP)
        y1 = SUGGESTION_Y
        x2 = x1 + btn_w
        y2 = y1 + SUGGESTION_H
        rects.append((word, x1, y1, x2, y2))
    return rects


def draw_suggestions(frame, sug_rects, hovered_sug, dwell_sug, dwell_progress):
    """
    Render the suggestion buttons above the keyboard.
    """
    overlay = frame.copy()
    for word, x1, y1, x2, y2 in sug_rects:
        if word == dwell_sug and dwell_progress > 0:
            t = dwell_progress
            color = (
                int(SUGGESTION_COLOR[0] * (1-t) + SUGGESTION_HOVER[0] * t),
                int(SUGGESTION_COLOR[1] * (1-t) + SUGGESTION_HOVER[1] * t),
                int(SUGGESTION_COLOR[2] * (1-t) + SUGGESTION_HOVER[2] * t),
            )
        elif word == hovered_sug:
            color = (55, 45, 65)
        else:
            color = SUGGESTION_COLOR

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), SUGGESTION_ACCENT, 1)

        # Progress bar 
        if word == dwell_sug and dwell_progress > 0:
            bar_w = int((x2 - x1 - 4) * dwell_progress)
            cv2.rectangle(overlay, (x1 + 2, y2 - 4), (x1 + 2 + bar_w, y2 - 1),SUGGESTION_ACCENT, -1)

        display = word.upper()
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2 - 2
        cv2.putText(overlay, display, (tx, ty),cv2.FONT_HERSHEY_SIMPLEX, 0.5, SUGGESTION_ACCENT, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)


def build_key_rects(layout):
    keys = []
    for row_idx, row in enumerate(layout):
        y1 = KB_ORIGIN_Y + row_idx * (KEY_H + KEY_MARGIN)
        y2 = y1 + KEY_H
        if row_idx == len(layout) - 1:
            wide_w = (len(layout[0]) * (KEY_W + KEY_MARGIN) - KEY_MARGIN) // len(row)
            for col_idx, label in enumerate(row):
                x1 = KB_ORIGIN_X + col_idx * (wide_w + KEY_MARGIN)
                x2 = x1 + wide_w
                keys.append((label, x1, y1, x2, y2))
        else:
            for col_idx, label in enumerate(row):
                x1 = KB_ORIGIN_X + col_idx * (KEY_W + KEY_MARGIN)
                x2 = x1 + KEY_W
                keys.append((label, x1, y1, x2, y2))
    return keys


def draw_keyboard(frame, keys, hovered_key, pressed_key, dwell_key, dwell_progress):
  
    overlay = frame.copy()
    for label, x1, y1, x2, y2 in keys:
        if label == pressed_key:
            color = PRESS_COLOR
        elif label == dwell_key and dwell_progress > 0:
            t = dwell_progress
            color = (
                int(KEY_BG_COLOR[0] * (1-t) + HOVER_COLOR[0] * t),
                int(KEY_BG_COLOR[1] * (1-t) + HOVER_COLOR[1] * t),
                int(KEY_BG_COLOR[2] * (1-t) + HOVER_COLOR[2] * t),
            )
        elif label == hovered_key:
            color = (45, 45, 60)
        else:
            color = KEY_BG_COLOR

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 140), 1)

        # If this key is being dwelled, draw a progress bar at the bottom of the key
        if label == dwell_key and dwell_progress > 0:
            bar_w = int((x2 - x1 - 6) * dwell_progress)
            bar_color = (0, int(200 * dwell_progress), 255)
            cv2.rectangle(overlay, (x1 + 3, y2 - 6), (x1 + 3 + bar_w, y2 - 2), bar_color, -1)

        display = "SPC" if label == "SPACE" else ("ENT" if label == "ENTER" else label)
        font_scale = 0.55 if len(display) == 1 else 0.42
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(overlay, display, (tx, ty),cv2.FONT_HERSHEY_SIMPLEX, font_scale, KEY_TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)


def count_extended_fingers(hand_landmarks, w, h):
    tips    = [4, 8, 12, 16, 20]
    bases   = [2, 6, 10, 14, 18]
    count = 0
    for tip_id, base_id in zip(tips, bases):
        tip  = hand_landmarks.landmark[tip_id]
        base = hand_landmarks.landmark[base_id]
        if tip.y < base.y:
            count += 1
    return count


#  Main driver function
def run_keyboard():
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for Hand Keyboard.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    layout_idx   = 0
    layouts      = [LAYOUT_LETTERS, LAYOUT_NUMBERS]
    keys         = build_key_rects(layouts[layout_idx])

    # Dwell state
    dwell_key        = None   
    dwell_start      = 0       
    dwell_progress   = 0.0     

    # Press state
    pressed_key      = None    
    press_time       = 0      
    last_fired_key   = None   
    last_fired_time  = 0      

    palm_open_last   = False
    palm_debounce_t  = 0

    typed_text = ""
    suggestions = []      
    sug_rects   = []      

    print(" Air Keyboard running — hover on a key to type. Press ESC to quit.")
    print("   Word suggestions appear as you type!")

    two_hand_count = 0  # counter for two-hand exit gesture

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Two-hand exit: 
        if (result.multi_hand_landmarks and len(result.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — returning to menu.")
                break
        else:
            two_hand_count = 0

        hovered_key = None
        finger_tip  = None
        now         = time.time()

        if pressed_key and now - press_time > 0.3:
            pressed_key = None

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                idx_tip = hand_lm.landmark[8]
                fx = int(idx_tip.x * w)
                fy = int(idx_tip.y * h)
                finger_tip = (fx, fy)

                n_fingers = count_extended_fingers(hand_lm, w, h)
                palm_open = n_fingers >= PALM_OPEN_THRESH
                if palm_open and not palm_open_last and (now - palm_debounce_t) > 0.8:
                    layout_idx = 1 - layout_idx
                    keys = build_key_rects(layouts[layout_idx])
                    palm_debounce_t = now
                    dwell_key = None  
                palm_open_last = palm_open

                for label, x1, y1, x2, y2 in keys:
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        hovered_key = label
                        break

                hovered_sug = None
                for word, x1, y1, x2, y2 in sug_rects:
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        hovered_sug = word
                        break

                dwell_target = None
                if hovered_sug:
                    dwell_target = "__SUG__" + hovered_sug
                elif hovered_key:
                    dwell_target = hovered_key

                if dwell_target is not None:
                    if dwell_target == dwell_key:
                        elapsed = now - dwell_start
                        dwell_progress = min(elapsed / DWELL_TIME, 1.0)

                        if dwell_progress >= 1.0:
                            if dwell_target != last_fired_key or (now - last_fired_time) > DWELL_COOLDOWN:

                                pressed_key     = dwell_target
                                press_time      = now
                                last_fired_key  = dwell_target
                                last_fired_time = now

                                if dwell_target.startswith("__SUG__"):
                                    selected_word = dwell_target[7:]  
                                    words_so_far = typed_text.split()
                                    if words_so_far and not typed_text.endswith(" "):
                                        words_so_far[-1] = selected_word.upper()
                                    else:
                                        words_so_far.append(selected_word.upper())

                                    typed_text = " ".join(words_so_far) + " "
                                    partial = typed_text.split()[-2] if len(typed_text.split()) >= 2 else selected_word.upper()
                                    pyautogui.typewrite(selected_word + " ", interval=0.02)

                                elif dwell_target == "SPACE":
                                    pyautogui.press("space")
                                    typed_text += " "
                                elif dwell_target == "ENTER":
                                    pyautogui.press("enter")
                                    typed_text += "\n"
                                elif dwell_target in ("⌫", "Backspace"):
                                    pyautogui.press("backspace")
                                    typed_text = typed_text[:-1]
                                else:
                                    pyautogui.press(dwell_target.lower())
                                    typed_text += dwell_target

                                suggestions = get_suggestions(typed_text)
                                sug_rects   = build_suggestion_rects(suggestions)

                            dwell_key      = None
                            dwell_start    = 0
                            dwell_progress = 0.0
                    else:
                        dwell_key      = dwell_target
                        dwell_start    = now
                        dwell_progress = 0.0
                else:
                    dwell_key      = None
                    dwell_start    = 0
                    dwell_progress = 0.0

                # draw fingertip 
                cv2.circle(frame, finger_tip, 12, (0, 255, 200), -1)
                cv2.circle(frame, finger_tip, 12, (255, 255, 255), 2)

                # draw progress ring around cursor when dwelling
                if dwell_progress > 0:
                    angle = int(360 * dwell_progress)
                    ring_color = (0, int(200 * dwell_progress), 255)
                    cv2.ellipse(frame, finger_tip, (22, 22), -90, 0, angle, ring_color, 3)

        else:
            
            dwell_key      = None
            dwell_start    = 0
            dwell_progress = 0.0
            palm_open_last = False

        suggestions = get_suggestions(typed_text)
        sug_rects   = build_suggestion_rects(suggestions)

        #draw suggestions
        if sug_rects:
            active_sug = None
            active_sug_progress = 0.0
            if dwell_key and dwell_key.startswith("__SUG__"):
                active_sug = dwell_key[7:]
                active_sug_progress = dwell_progress
            hovered_sug_name = None
            if result.multi_hand_landmarks and finger_tip:
                for word, x1, y1, x2, y2 in sug_rects:
                    if x1 <= finger_tip[0] <= x2 and y1 <= finger_tip[1] <= y2:
                        hovered_sug_name = word
                        break
            draw_suggestions(frame, sug_rects, hovered_sug_name, active_sug, active_sug_progress)

        kb_dwell_key = dwell_key if (dwell_key and not dwell_key.startswith("__SUG__")) else None
        draw_keyboard(frame, keys, hovered_key, pressed_key, kb_dwell_key, dwell_progress if kb_dwell_key else 0)

        layout_name = "ABC" if layout_idx == 0 else "123"
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 30), -1)
        cv2.putText(frame, f"Air Keyboard  [{layout_name}]  |  Hold = type  |  Suggestions auto-complete",(14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)

        preview = ">> " + typed_text[-55:]
        cv2.rectangle(frame, (0, SUGGESTION_Y - 44), (w, SUGGESTION_Y - 2), (20, 20, 40), -1)
        cv2.putText(frame, preview, (14, SUGGESTION_Y - 14),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 180), 1, cv2.LINE_AA)

        cv2.imshow("Air Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye! You typed:", repr(typed_text))


if __name__ == "__main__":
    run_keyboard()