import cv2 as cv
import face_recognition as fr
import numpy as np
import pickle
import os
import time

ENCODINGS_FILE     = "user_encodings.pkl"
FRAMES_TO_CAPTURE  = 15      
MATCH_THRESHOLD    = 0.5    
REGISTRATION_DELAY = 0.3    

def capture_face(frame):
    """
    Takes a BGR frame from OpenCV.
    Converts to RGB for face_recognition.
    Returns face locations and cropped face.
    Returns None, None if no face found.
    """
    # OpenCV gives BGR — face_recognition needs RGB
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    locations = fr.face_locations(rgb_frame)

    if len(locations) == 0:
        return None, None

    top, right, bottom, left = locations[0]
    face_crop = frame[top:bottom, left:right]

    return locations, face_crop


def register_user(cap):
    """
    Captures multiple frames of the user's face.
    Saves all encodings to disk as .pkl file.
    Only runs when no .pkl file exists (first time only).
    """
    print("\n[ REGISTRATION MODE ]")
    print("Look at the camera. Registration starts in 3 seconds...")
    time.sleep(3)

    encodings = []
    captured  = 0

    print(f"Capturing {FRAMES_TO_CAPTURE} frames of your face...")

    while captured < FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)

        # convert to RGB for face_recognition
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        locations, face_crop = capture_face(frame)

        # no face found in this frame
        if locations is None:
            cv.putText(frame, "No face detected — move closer",
                       (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
            cv.imshow("Registration", frame)
            cv.waitKey(1)
            continue

        # get encoding using RGB frame
        face_encodings = fr.face_encodings(rgb_frame, locations)

        if len(face_encodings) == 0:
            continue

        encodings.append(face_encodings[0])
        captured += 1

        # show progress on screen
        top, right, bottom, left = locations[0]
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.putText(frame, f"Capturing {captured}/{FRAMES_TO_CAPTURE}",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        cv.imshow("Registration", frame)
        cv.waitKey(1)

        # small delay so each frame is slightly different
        time.sleep(REGISTRATION_DELAY)

    # save all encodings to disk
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

    print(f"\n[ REGISTRATION COMPLETE ]")
    print(f"Saved {len(encodings)} encodings to {ENCODINGS_FILE}")
    print("Restart the program to login.\n")

    cv.destroyWindow("Registration")


def recognize_user(frame):
    """
    Compares face in current frame against saved encodings.
    Returns:
    - matched    (True or False)
    - distance   (float — how close, lower is better)
    - confidence (float — percentage 0 to 100)
    """
    # load saved encodings from disk
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings = pickle.load(f)

    # convert to RGB for face_recognition
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # find face location
    locations, _ = capture_face(frame)

    if locations is None:
        return False, None, None

    # get encoding using RGB frame
    face_encodings = fr.face_encodings(rgb_frame, locations)

    if len(face_encodings) == 0:
        return False, None, None

    unknown_encoding = face_encodings[0]

    # compare against every saved encoding
    distances = fr.face_distance(known_encodings, unknown_encoding)
    matches   = fr.compare_faces(known_encodings, unknown_encoding,
                                 tolerance=MATCH_THRESHOLD)

    # find the best match out of all saved encodings
    best_index    = np.argmin(distances)
    best_distance = distances[best_index]
    best_match    = matches[best_index]
    confidence    = round((1 - best_distance) * 100, 1)

    return best_match, best_distance, confidence


def run_auth():
    """
    Master function — runs the full authentication flow.
    Returns True if authentication passed.
    Returns False if denied or timed out.
    """
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)   # CAP_DSHOW for Windows

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return False

    # first time — no encodings file exists yet
    if not os.path.exists(ENCODINGS_FILE):
        print("No registered user found.")
        register_user(cap)
        cap.release()
        cv.destroyAllWindows()
        return False   # restart needed after registration

    # every time after — login mode
    print("\n[ AUTHENTICATION MODE ]")
    print("Look at the camera...")

    auth_result = False
    start_time  = time.time()
    TIMEOUT     = 10   # seconds before giving up

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)

        # check timeout
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT:
            print("Authentication timed out.")
            break

        # try to recognize current face
        matched, distance, confidence = recognize_user(frame)

        # draw UI on frame
        locations, _ = capture_face(frame)

        if locations is not None:
            top, right, bottom, left = locations[0]

            if matched:
                color = (0, 255, 0)    # green = match
                label = f"GRANTED  {confidence}%"
            else:
                color = (0, 0, 255)    # red = no match
                label = f"DENIED  {confidence}%" if confidence else "DENIED"

            # box around face
            cv.rectangle(frame, (left, top), (right, bottom), color, 2)

            # label below box
            cv.rectangle(frame, (left, bottom), (right, bottom + 35),
                         color, cv.FILLED)
            cv.putText(frame, label, (left + 6, bottom + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # show countdown timer
        cv.putText(frame, f"Time: {int(TIMEOUT - elapsed)}s",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        cv.imshow("Face Authentication", frame)
        cv.waitKey(1)

        # if matched hold screen for 1.5 seconds then proceed
        if matched:
            print(f"\n[ ACCESS GRANTED ] Confidence: {confidence}%")
            time.sleep(1.5)
            auth_result = True
            break

    cap.release()
    cv.destroyAllWindows()
    return auth_result


if __name__ == "__main__":
    result = run_auth()

    if result:
        print("Proceeding to main menu...")
        # main menu will be called here later
    else:
        print("Authentication failed. Exiting.")