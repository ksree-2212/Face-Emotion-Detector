"""
detect_live.py
---------------
Real-time face detection + emotion classification from a webcam feed.

Pipeline per frame:
    1. Grab frame from webcam
    2. Convert to grayscale
    3. Detect faces with Haar Cascade
    4. For each face: crop -> resize to 48x48 -> normalize -> run through CNN
    5. Overlay bounding box + predicted emotion + confidence on the frame

Press 'q' to quit.
"""

import os
import cv2
import numpy as np

from model_def import EMOTION_CLASSES, IMG_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_cnn.h5")
CONFIDENCE_THRESHOLD = 0.45  # below this, label as "Uncertain" rather than force a guess

# Haar cascade ships with opencv-python, no separate download needed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No trained model found at {MODEL_PATH}.")
        print("Face detection will still run, but emotion labels will show 'No model'.")
        print("Run `python src/train_model.py` first to train one.")
        return None
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


def predict_emotion(model, face_roi_gray):
    face = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # shape (1, 48, 48, 1)
    preds = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return EMOTION_CLASSES[idx], confidence


def main():
    model = load_emotion_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            if model is not None:
                roi_gray = gray[y:y + h, x:x + w]
                label, conf = predict_emotion(model, roi_gray)
                text = f"{label} ({conf*100:.0f}%)" if conf >= CONFIDENCE_THRESHOLD else "Uncertain"
            else:
                text = "No model loaded"

            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
            )

        cv2.imshow("Face & Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
