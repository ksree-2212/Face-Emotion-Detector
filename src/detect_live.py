"""
Real-time webcam face detection + emotion classification.
Predictions are averaged over the last few frames per face, which removes the
label flicker you get from classifying every frame independently.
Press 'q' to quit.
"""
from collections import deque

import cv2
import numpy as np

from emotion_utils import detect_faces, format_label, load_emotion_model, predict_probs

SMOOTH_FRAMES = 8
MATCH_DIST = 80  # px: max centre movement to treat a box as the same face


def main():
    model, labels = load_emotion_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    tracks = []  # each: {"c": (cx, cy), "hist": deque of prob vectors}
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_tracks = []
        for (x, y, w, h) in detect_faces(gray):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            text = "No model loaded"
            if model is not None:
                c = (x + w / 2, y + h / 2)
                tr = next((t for t in tracks
                           if np.hypot(t["c"][0] - c[0], t["c"][1] - c[1]) < MATCH_DIST), None)
                if tr is None:
                    tr = {"hist": deque(maxlen=SMOOTH_FRAMES)}
                tr["c"] = c
                tr["hist"].append(predict_probs(model, gray[y:y + h, x:x + w]))
                new_tracks.append(tr)
                text = format_label(labels, np.mean(tr["hist"], axis=0))
            cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        tracks = new_tracks

        cv2.imshow("Face & Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
