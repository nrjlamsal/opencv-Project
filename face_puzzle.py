"""
🧩 Face Puzzle — Live Sliding Puzzle with Your Face!

Displays a 3*3 sliding puzzle built from your LIVE webcam feed.
Each tile shows a real-time portion of the camera image, rearranged
by the puzzle shuffle.  Tiles are numbered 1-8 so you can track them.

Controls:
    - Arrow keys  = slide tiles
    - 'r' = reshuffle the puzzle
    - 'q'  = quit
    - Show two hands = return to menu



The puzzle is shuffled via valid random moves to guarantee solvability.
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import random
import time
import math

# for two-hand exit
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# confuguration
ROWS, COLS   = 3, 3
TOTAL_TILES  = ROWS * COLS
PUZZLE_SIZE  = 480          
PREVIEW_W    = 160          # camera  width
PREVIEW_H    = 120          # camera  height
CANVAS_W     = PUZZLE_SIZE + PREVIEW_W + 40   
CANVAS_H     = PUZZLE_SIZE + 100               

#colors
BG_COLOR      = (18, 18, 22)
GRID_COLOR    = (200, 200, 210)
EMPTY_COLOR   = (30, 30, 38)
BADGE_BG      = (0, 0, 0)
BADGE_TEXT     = (255, 255, 255)
ACCENT        = (0, 220, 255)
WIN_COLOR     = (0, 220, 100)
TEXT_DIM      = (120, 120, 130)


def get_neighbors(index):
    neighbors = []
    r, c = index // COLS, index % COLS
    if r > 0:
        neighbors.append(index - COLS)
    if r < ROWS - 1:
        neighbors.append(index + COLS)
    if c > 0:
        neighbors.append(index - 1)
    if c < COLS - 1:
        neighbors.append(index + 1)
    return neighbors


def create_shuffle():
    """
    Create a shuffled tile mapping via valid moves 
    """
    tile_map = list(range(TOTAL_TILES))   # identity=[0,1,2,...,8]
    empty_pos = TOTAL_TILES - 1           

    prev = -1
    for _ in range(300):
        neighbors = get_neighbors(empty_pos)
        choices = [n for n in neighbors if n != prev]
        if not choices:
            choices = neighbors
        n = random.choice(choices)
        tile_map[empty_pos], tile_map[n] = tile_map[n], tile_map[empty_pos]
        prev = empty_pos
        empty_pos = n

    return tile_map, empty_pos


def is_solved(tile_map, empty_pos):
    """Check if the puzzle is in the solved state"""
    for i in range(TOTAL_TILES):
        if i == empty_pos:
            continue
        if tile_map[i] != i:
            return False
    return True


def draw_rounded_rect(img, x, y, w, h, r, color, alpha=1.0):
    """Draw a filled rectangle with rounded corners"""
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


# main driver function

def run_puzzle():
   
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera for Face Puzzle.")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    cv.namedWindow("Face Puzzle", cv.WINDOW_NORMAL)
    cv.setWindowProperty("Face Puzzle", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Tile dimensions
    tile_w = PUZZLE_SIZE // COLS
    tile_h = PUZZLE_SIZE // ROWS

    # Puzzle offset on canvas 
    puzzle_x = 20
    puzzle_y = 60

    # initial shuffle
    tile_map, empty_pos = create_shuffle()
    solved = False

    print("\n==========================================")
    print("  FACE PUZZLE — Live Sliding Puzzle")
    print("==========================================")
    print("  Arrow keys = move tiles")
    print("  'r' = reshuffle  |  'q' = quit")
    print("  Show two hands = return to menu")
    print("==========================================\n")

    two_hand_count = 0  #  counter for two-hand exit gesture

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # Twohand exit gesture 
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        hand_result = hands_model.process(rgb)
        if (hand_result.multi_hand_landmarks
                and len(hand_result.multi_hand_landmarks) >= 2):
            two_hand_count += 1
            if two_hand_count >= 15:
                print("  [EXIT] Two hands held — returning to menu.")
                break
        else:
            two_hand_count = 0

        puzzle_frame = cv.resize(frame, (PUZZLE_SIZE, PUZZLE_SIZE))

        # Build canvas 
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR

    
        cv.putText(canvas, "FACE PUZZLE",(puzzle_x + PUZZLE_SIZE // 2 - 100, 40),cv.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2, cv.LINE_AA)

        # draw puzzle tiles
        now = time.time()

        for pos in range(TOTAL_TILES):
            row = pos // COLS
            col = pos % COLS

            # Display coordinates
            dx = puzzle_x + col * tile_w
            dy = puzzle_y + row * tile_h

            if pos == empty_pos and not solved:
                # Empty slot 
                cv.rectangle(canvas, (dx, dy), (dx + tile_w, dy + tile_h), EMPTY_COLOR, -1)
                cv.line(canvas, (dx, dy), (dx + tile_w, dy + tile_h),(40, 40, 50), 1)
                cv.line(canvas, (dx + tile_w, dy), (dx, dy + tile_h),(40, 40, 50), 1)
                continue

            # Source tile index
            src = tile_map[pos]
            src_row = src // COLS
            src_col = src % COLS

            sx1 = src_col * tile_w
            sy1 = src_row * tile_h
            sx2 = sx1 + tile_w
            sy2 = sy1 + tile_h
            tile_img = puzzle_frame[sy1:sy2, sx1:sx2]

            # Place tile on canvas
            canvas[dy:dy + tile_h, dx:dx + tile_w] = tile_img

            #  Grid border
            cv.rectangle(canvas, (dx, dy), (dx + tile_w - 1, dy + tile_h - 1),GRID_COLOR, 2)

            #Tile number badge
            tile_num = src + 1   
            badge_size = 28
            badge_overlay = canvas.copy()
            cv.rectangle(badge_overlay,(dx + 4, dy + 4),(dx + 4 + badge_size, dy + 4 + badge_size),BADGE_BG, -1)
            cv.addWeighted(badge_overlay, 0.7, canvas, 0.3, 0, canvas)

            cv.rectangle(canvas,(dx + 4, dy + 4),(dx + 4 + badge_size, dy + 4 + badge_size), ACCENT, 1)

            num_text = str(tile_num)
            (tw, th), _ = cv.getTextSize(num_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = dx + 4 + (badge_size - tw) // 2
            ty = dy + 4 + (badge_size + th) // 2
            cv.putText(canvas, num_text, (tx, ty),cv.FONT_HERSHEY_SIMPLEX, 0.6, BADGE_TEXT, 2, cv.LINE_AA)

            # If this tile is in its correct position, show a green dot
            if tile_map[pos] == pos:
                cv.circle(canvas, (dx + tile_w - 14, dy + 14), 6, WIN_COLOR, -1)
                cv.circle(canvas, (dx + tile_w - 14, dy + 14), 6, (255, 255, 255), 1)

        # reference image
        preview_x = puzzle_x + PUZZLE_SIZE + 15
        preview_y = puzzle_y

        cv.putText(canvas, "LIVE VIEW", (preview_x, preview_y - 8),cv.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1)

        # Resize frame for preview
        preview = cv.resize(frame, (PREVIEW_W, PREVIEW_H))
        canvas[preview_y:preview_y + PREVIEW_H, preview_x:preview_x + PREVIEW_W] = preview

        # show border
        cv.rectangle(canvas, (preview_x - 2, preview_y - 2), (preview_x + PREVIEW_W + 2, preview_y + PREVIEW_H + 2), (60, 60, 70), 2)

        controls_y = preview_y + PREVIEW_H + 25
        controls = [
            "Arrow keys = move",
            "r = reshuffle",
            "q = quit",
            "Two hands = menu",
        ]
        for i, line in enumerate(controls):
            cv.putText(canvas, line,(preview_x + 8, controls_y + i * 22),cv.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 80), 1)

        if solved:
            pulse = abs(math.sin(now * 3))
            glow = int(150 + 105 * pulse)
            cv.rectangle(canvas, (puzzle_x - 3, puzzle_y - 3), (puzzle_x + PUZZLE_SIZE + 3, puzzle_y + PUZZLE_SIZE + 3),(0, glow, int(glow * 0.6)), 3)

            banner_y = puzzle_y + PUZZLE_SIZE // 2 - 25
            win_overlay = canvas.copy()
            cv.rectangle(win_overlay,(puzzle_x, banner_y),(puzzle_x + PUZZLE_SIZE, banner_y + 50),(0, 180, 80), -1)
            cv.addWeighted(win_overlay, 0.8, canvas, 0.2, 0, canvas)

            win_text = "PUZZLE SOLVED!"
            (wt, _), _ = cv.getTextSize(win_text, cv.FONT_HERSHEY_DUPLEX, 0.85, 2)
            cv.putText(canvas, win_text(puzzle_x + (PUZZLE_SIZE - wt) // 2, banner_y + 35),cv.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv.LINE_AA)


        cv.imshow("Face Puzzle", canvas)
        key = cv.waitKeyEx(1)

        if key == ord('q') or key == 27:
            break

        if key == ord('r'):
            tile_map, empty_pos = create_shuffle()
            solved = False
            continue

        if solved:
            continue   

        LEFT  = 2424832
        UP    = 2490368
        RIGHT = 2555904
        DOWN  = 2621440

        moved = False

        if key == LEFT:
            if empty_pos % COLS < COLS - 1:
                swap = empty_pos + 1
                tile_map[empty_pos], tile_map[swap] = tile_map[swap], tile_map[empty_pos]
                empty_pos = swap
                moved = True

        elif key == RIGHT:
            if empty_pos % COLS > 0:
                swap = empty_pos - 1
                tile_map[empty_pos], tile_map[swap] = tile_map[swap], tile_map[empty_pos]
                empty_pos = swap
                moved = True

        elif key == UP:
            if empty_pos // COLS < ROWS - 1:
                swap = empty_pos + COLS
                tile_map[empty_pos], tile_map[swap] = tile_map[swap], tile_map[empty_pos]
                empty_pos = swap
                moved = True

        elif key == DOWN:
            if empty_pos // COLS > 0:
                swap = empty_pos - COLS
                tile_map[empty_pos], tile_map[swap] = tile_map[swap], tile_map[empty_pos]
                empty_pos = swap
                moved = True

        if moved:
            if is_solved(tile_map, empty_pos):
                solved = True
                print("  🎉 PUZZLE SOLVED!")

    cap.release()
    cv.destroyWindow("Face Puzzle")
    print("Face Puzzle closed.")


if __name__ == "__main__":
    run_puzzle()