"""Shared inference helpers for detect_live.py and detect_image.py."""
import os

import cv2
import numpy as np

from model_def import IMG_SIZE, MODELS_DIR, load_class_names

MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn.h5")
CONFIDENCE_THRESHOLD = 0.45

# Haar cascade = classical, pre-built face LOCALISER shipped with OpenCV (not an emotion model).
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] No trained model at {MODEL_PATH}. Run: python src/train_model.py")
        return None, None
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH), load_class_names()


def detect_faces(gray):
    return FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))


def predict_probs(model, face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return model.predict(face[None, ..., None], verbose=0)[0]


def format_label(labels, probs):
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return (f"{labels[idx]} ({conf * 100:.0f}%)" if conf >= CONFIDENCE_THRESHOLD else "Uncertain")
