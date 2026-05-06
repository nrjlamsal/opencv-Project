# Contactless HCI System

A contactless human-computer interaction system built with Python, OpenCV, and MediaPipe. You can draw, type, play games, and control your mouse — all without touching your keyboard or mouse.

Built for CS 2100 at the University of Idaho.

## What it does

- **Face Authentication** — registers your face on first run, then requires face login every time
- **Air Canvas** — draw on screen by moving your index finger in the air
- **Hand Keyboard** — virtual keyboard, hover over keys to type
- **Blink Morse Code** — blink your eyes to spell out words in morse code
- **Rock Paper Scissors** — play RPS against the computer using hand gestures
- **Hand Mouse** — move your cursor and click using hand gestures
- **Face Puzzle** — sliding puzzle that uses your live webcam face as tiles

Everything is accessed through a gesture-controlled main menu. Point at a tile, pinch to select.

## Setup

You need Python 3.11 and a webcam.

```
pip install opencv-python mediapipe face-recognition numpy pyautogui
```

face-recognition also needs dlib, which can be tricky to install on Windows. If you run into issues, try:
```
pip install cmake
pip install dlib
pip install face-recognition
```

## How to run

```
cd real_project
python main.py
```

On first run it will ask you to register your face (looks at you through the camera for a few seconds). After that it goes straight to login.

## Controls

- **Main menu**: point your index finger at a tile, pinch (thumb + index) and hold to select. Or just press 1-6 on keyboard.
- **Exit any module**: show both hands to the camera for about half a second
- **Quit app**: show both hands from the main menu, or press Q

## Files

| File | What it does |
|------|-------------|
| main.py | Entry point, main menu, launches modules |
| face_auth.py | Face registration and login |
| canvas.py | Air canvas drawing |
| handkeyboard.py | Virtual keyboard with word prediction |
| blink_morse.py | Blink morse code |
| rps_game.py | Rock paper scissors game |
| hand_mouse.py | Cursor control |
| face_puzzle.py | Sliding puzzle |
| handmodule.py | Shared hand detection wrapper |
| facemodule.py | Shared face detection wrapper |

## Notes

- The face encoding file (`user_encodings.pkl`) is in .gitignore so it won't get pushed to github
- If a module fails to import (missing dependency), it just shows as unavailable in the menu instead of crashing
- Camera needs to be available — if another app is using it, the program will tell you

## Author

Niraj Lamsal — University of Idaho
