"""
detect_image.py
----------------
Same detection + emotion-classification pipeline as detect_live.py,
but runs on a single static image instead of a webcam feed.
Useful for testing/demoing the pipeline in environments without a camera
(e.g. this project's sandbox, CI, or a screen-recorded demo for a resume).

Usage:
    python src/detect_image.py --image path/to/photo.jpg --output path/to/result.jpg
"""

import argparse
import os
import cv2
import numpy as np

from model_def import EMOTION_CLASSES, IMG_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_cnn.h5")
CONFIDENCE_THRESHOLD = 0.45

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


def predict_emotion(model, face_roi_gray):
    face = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    preds = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(preds))
    return EMOTION_CLASSES[idx], float(preds[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Path to save annotated output image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    model = load_emotion_model()
    if model is None:
        print("[WARN] No trained model found — will only draw face boxes, no emotion label.")

    frame = cv2.imread(args.image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    print(f"Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        if model is not None:
            roi_gray = gray[y:y + h, x:x + w]
            label, conf = predict_emotion(model, roi_gray)
            text = f"{label} ({conf*100:.0f}%)" if conf >= CONFIDENCE_THRESHOLD else "Uncertain"
            print(f"  Face at ({x},{y},{w},{h}): {text}")
        else:
            text = "Face"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    out_path = args.output or (os.path.splitext(args.image)[0] + "_annotated.jpg")
    cv2.imwrite(out_path, frame)
    print(f"Saved annotated image to: {out_path}")


if __name__ == "__main__":
    main()
