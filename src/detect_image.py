"""
Run face detection + emotion classification on a single image.
Usage: python src/detect_image.py --image photo.jpg [--output result.jpg]
"""
import argparse
import os

import cv2

from emotion_utils import detect_faces, format_label, load_emotion_model, predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] Could not read image: {args.image}")
        return
    model, labels = load_emotion_model()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    print(f"Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        text = "Face"
        if model is not None:
            text = format_label(labels, predict_probs(model, gray[y:y + h, x:x + w]))
            print(f"  Face at ({x},{y},{w},{h}): {text}")
        cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    out = args.output or (os.path.splitext(args.image)[0] + "_annotated.jpg")
    cv2.imwrite(out, frame)
    print(f"Saved annotated image to: {out}")


if __name__ == "__main__":
    main()
