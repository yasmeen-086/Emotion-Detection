import cv2
from deepface import DeepFace
import numpy as np


CAM_INDEX = 0               
WINDOW_NAME = "Emotion Detector (press q to quit)"
SCALE = 1.0                  
MIN_FACE_SIZE = (50, 50)     

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open webcam (index {CAM_INDEX}). Try changing CAM_INDEX.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade xml for face detection.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if SCALE != 1.0:#scling for speed
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    else:
        small = frame

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # detect faces (returns x,y,w,h) on the scaled frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )

    if len(faces) == 0:
        # optionally show a message when no face detected
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # map coordinates back to original frame if scaled
            if SCALE != 1.0:
                inv = 1.0 / SCALE
                x0 = int(x * inv)
                y0 = int(y * inv)
                w0 = int(w * inv)
                h0 = int(h * inv)
            else:
                x0, y0, w0, h0 = int(x), int(y), int(w), int(h)

            # ensure coords are inside frame
            x1 = max(0, x0)
            y1 = max(0, y0)
            x2 = min(frame.shape[1], x0 + w0)
            y2 = min(frame.shape[0], y0 + h0)

            # crop face region and convert BGR->RGB for DeepFace
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            try:

                analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                if isinstance(analysis, list) and len(analysis) > 0:
                    analysis = analysis[0]
                dominant_emotion = analysis.get("dominant_emotion", "Unknown")
                score = None
                emotions = analysis.get("emotion", None)
                if isinstance(emotions, dict):
                    score = emotions.get(dominant_emotion, None)
                label = f"{dominant_emotion}" + (f" ({score:.2f})" if score is not None else "")
            except Exception:
                label = "Emotion: Error"

            # draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            px, py = x1, y1 - 10 if y1 - 10 > 20 else y1 + lh + 10
            cv2.rectangle(frame, (px, py - lh - 6), (px + lw + 6, py + 6), (0, 0, 0), -1)
            cv2.putText(frame, label, (px + 3, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
