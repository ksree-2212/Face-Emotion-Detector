"""
evaluate_model.py
-----------------
Evaluates the existing .h5 on data/fer2013/test without training.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from emotion_utils import load_emotion_model
from model_def import IMG_SIZE

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data", "fer2013")

def main():
    model, names = load_emotion_model()
    if model is None:
        return
        
    plain_test = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = plain_test.flow_from_directory(
        os.path.join(DATA_DIR, "test"), 
        target_size=(IMG_SIZE, IMG_SIZE), 
        color_mode="grayscale",
        batch_size=64, 
        class_mode="categorical",
        shuffle=False
    )

    print("Class order:", names)

    probs = model.predict(test_gen, verbose=0)
    y_pred, y_true = probs.argmax(1), test_gen.classes
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    print(f"\nTEST accuracy: {acc:.4f}\n{'class':<10}{'prec':>7}{'recall':>8}{'support':>9}")
    for i, name in enumerate(names):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        print(f"{name:<10}{prec:>7.2f}{rec:>8.2f}{cm[i].sum():>9}")

if __name__ == "__main__":
    main()
