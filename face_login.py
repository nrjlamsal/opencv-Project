import face_recognition
import cv2 as cv
import numpy as np
import os

# ─── CONFIG ───────────────────────────────────────────────
ENCODING_PATH        = "face_data/encoding.npy"
PHOTOS_NEEDED        = 5
MATCH_THRESHOLD      = 0.5    # lower = stricter. 0.4 strict, 0.6 lenient
TEXTURE_THRESHOLD    = 100.0  # lower it to 50-60 if real face keeps failing
DETECT_EVERY_N       = 3      # run face detection every N frames (reduces lag)
FRAME_SCALE          = 0.5    # shrink frame to this size for detection only


# ─── STEP 1: CAPTURE FACE ─────────────────────────────────
def capture_face(frame):
    """
    Find face in frame.
    Resizes frame to half size for faster detection.
    Scales coordinates back to full size.
    Uses HOG model (faster than CNN, accurate enough).
    Returns (face_location, face_crop) or (None, None) if no face found.
    """
    # shrink frame for faster detection
    small = cv.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb   = cv.cvtColor(small, cv.COLOR_BGR2RGB)

    # hog is much faster than default cnn model
    locations = face_recognition.face_locations(rgb, model="hog")

    if not locations:
        return None, None

    # scale coordinates back to full frame size
    top, right, bottom, left = locations[0]
    top    = int(top    / FRAME_SCALE)
    right  = int(right  / FRAME_SCALE)
    bottom = int(bottom / FRAME_SCALE)
    left   = int(left   / FRAME_SCALE)

    face_crop = frame[top:bottom, left:right]
    return (top, right, bottom, left), face_crop


# ─── STEP 2: TEXTURE CHECK (ANTI SPOOF) ───────────────────
def check_texture(face_crop):
    """
    Real faces have skin texture — photos are flat and smooth.
    Laplacian measures how much detail is in the image.
    High variance = real face. Low variance = flat printed photo.
    Returns True if real, False if likely fake.
    """
    if face_crop is None or face_crop.size == 0:
        return False

    gray      = cv.cvtColor(face_crop, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    variance  = laplacian.var()

    print(f"Texture variance: {variance:.2f}  (need > {TEXTURE_THRESHOLD})")
    return variance > TEXTURE_THRESHOLD


# ─── STEP 3: REGISTRATION ─────────────────────────────────
def register_user(cap):
    """
    First time setup.
    Captures 5 photos, computes encodings, averages them, saves as .npy file.
    """
    print("\n--- REGISTRATION MODE ---")
    print("Look at the camera. Press SPACE to capture each photo.")

    os.makedirs("face_data", exist_ok=True)

    encodings_collected = []
    count       = 0
    frame_count = 0
    location    = None

    while count < PHOTOS_NEEDED:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        frame_count += 1

        # only run detection every N frames
        if frame_count % DETECT_EVERY_N == 0:
            location, _ = capture_face(frame)

        display = frame.copy()
        cv.putText(display,
                   f"Photo {count}/{PHOTOS_NEEDED}  —  Press SPACE to capture",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if location:
            top, right, bottom, left = location
            cv.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(display, "Face detected — ready to capture",
                       (left, top - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.55, (0, 255, 0), 2)
        else:
            cv.putText(display, "No face found — move closer",
                       (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow("Registration", display)
        key = cv.waitKey(1) & 0xFF

        if key == 32 and location:  # SPACE pressed and face found
            rgb      = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb, [location])

            if encoding:
                encodings_collected.append(encoding[0])
                count += 1
                print(f"  Captured photo {count}/{PHOTOS_NEEDED}")

                # flash green border to confirm capture
                confirm = display.copy()
                cv.rectangle(confirm, (left, top), (right, bottom), (0, 255, 0), 6)
                cv.putText(confirm, f"Captured {count}/{PHOTOS_NEEDED}",
                           (left, bottom + 25), cv.FONT_HERSHEY_SIMPLEX,
                           0.65, (0, 255, 0), 2)
                cv.imshow("Registration", confirm)
                cv.waitKey(400)

        if key == ord('q'):
            print("Registration cancelled.")
            cv.destroyAllWindows()
            return False

    # average all 5 encodings into one single array
    final_encoding = np.mean(encodings_collected, axis=0)

    # save as .npy
    np.save(ENCODING_PATH, final_encoding)
    print(f"\nRegistration complete. Encoding saved to {ENCODING_PATH}")
    print(f"Encoding shape: {final_encoding.shape}")  # should be (128,)

    cv.destroyWindow("Registration")
    return True


# ─── STEP 4: LOAD SAVED ENCODING ──────────────────────────
def load_encoding():
    """
    Load the saved face encoding from disk.
    Returns numpy array of shape (128,) or None if file not found.
    """
    if not os.path.exists(ENCODING_PATH):
        return None

    encoding = np.load(ENCODING_PATH)
    print(f"Encoding loaded from {ENCODING_PATH}  shape: {encoding.shape}")
    return encoding


# ─── STEP 5: RECOGNIZE FACE ───────────────────────────────
def recognize_face(frame, location, saved_encoding):
    """
    Compare current face to saved encoding.
    Returns (match: bool, distance: float)
    Lower distance = more similar. Under MATCH_THRESHOLD = same person.
    """
    rgb       = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb, [location])

    if not encodings:
        return False, 1.0

    distance = face_recognition.face_distance([saved_encoding], encodings[0])[0]
    match    = distance < MATCH_THRESHOLD

    print(f"Face distance: {distance:.3f}  (need < {MATCH_THRESHOLD})")
    return match, distance


# ─── MAIN AUTH FUNCTION ────────────────────────────────────
def run_auth():
    """
    Master function. Call this from main.py.
    Returns True if authenticated, exits program if failed.
    """
    cap = cv.VideoCapture(0)

    # check if user is already registered
    saved_encoding = load_encoding()

    if saved_encoding is None:
        print("No face data found. Starting registration...")
        success = register_user(cap)
        if not success:
            cap.release()
            return False
        saved_encoding = load_encoding()

    # ── LOGIN MODE ──
    print("\n--- LOGIN MODE ---")
    print("Look at the camera...")

    MAX_ATTEMPTS = 100  # frames to try before giving up
    attempt      = 0
    frame_count  = 0
    location     = None
    face_crop    = None

    while attempt < MAX_ATTEMPTS:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        frame_count += 1

        # only run face detection every N frames
        if frame_count % DETECT_EVERY_N == 0:
            location, face_crop = capture_face(frame)
            attempt += 1  # count detection attempts not total frames

        # always draw the display every frame for smooth video
        display = frame.copy()
        cv.putText(display, f"Authenticating...  ({attempt}/{MAX_ATTEMPTS})",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if location is None:
            cv.putText(display, "No face detected — move closer",
                       (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv.imshow("Authentication", display)
            cv.waitKey(1)
            continue

        # draw last known face box every frame
        top, right, bottom, left = location
        cv.rectangle(display, (left, top), (right, bottom), (255, 255, 0), 2)

        # only run checks on detection frames
        if frame_count % DETECT_EVERY_N == 0:

            # ── ANTI SPOOF: TEXTURE CHECK ──
            real = check_texture(face_crop)

            if not real:
                cv.putText(display, "SPOOF DETECTED — ACCESS DENIED",
                           (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.rectangle(display, (left, top), (right, bottom), (0, 0, 255), 3)
                cv.imshow("Authentication", display)
                cv.waitKey(2000)
                print("Spoof detected. Exiting.")
                break

            # ── FACE RECOGNITION ──
            match, distance = recognize_face(frame, location, saved_encoding)

            if match:
                confidence = int((1 - distance) * 100)
                cv.putText(display, f"ACCESS GRANTED  ({confidence}% match)",
                           (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 3)
                cv.imshow("Authentication", display)
                cv.waitKey(1500)
                cap.release()
                cv.destroyAllWindows()
                print("Authentication successful.")
                return True

            else:
                cv.putText(display, f"Unknown face  (distance: {distance:.2f})",
                           (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow("Authentication", display)
        cv.waitKey(1)

    # all attempts exhausted
    cap.release()
    cv.destroyAllWindows()
    print("Authentication failed. Exiting.")
    return False


# ─── RUN STANDALONE FOR TESTING ───────────────────────────
if __name__ == "__main__":
    result = run_auth()
    if result:
        print("You are in. Main program would start here.")
    else:
        print("Access denied. Exiting.")
        exit()