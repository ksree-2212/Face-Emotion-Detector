"""
train_model.py
--------------
Trains the emotion CNN from scratch on FER-2013.

Improvements over the original script:
  * Held-out TEST set is only used for the final report (no more picking the best
    epoch on the same data you report accuracy on). A 10% validation split of the
    training folder is used for checkpointing / early stopping.
  * Dampened class weights so the tiny 'disgust' class (436 imgs) is not ignored.
  * Saves models/class_names.json  -> fixes the train/inference label-order mismatch.
  * Prints per-class precision/recall, saves confusion matrix + training curves
    + metrics.json (handy material for the case-study report).

Usage:  python src/train_model.py --epochs 60
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_def import CLASS_NAMES_PATH, IMG_SIZE, MODELS_DIR, build_emotion_cnn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data", "fer2013")
MODEL_OUT = os.path.join(MODELS_DIR, "emotion_cnn.h5")
SEED = 42


def get_generators(batch_size):
    common = dict(target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale",
                  batch_size=batch_size, class_mode="categorical")
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=12, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True,
        brightness_range=(0.8, 1.2), validation_split=0.1)
    plain = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
    plain_test = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(DATA_DIR, "train")
    train_gen = train_aug.flow_from_directory(train_dir, subset="training", shuffle=True,
                                              seed=SEED, **common)
    val_gen = plain.flow_from_directory(train_dir, subset="validation", shuffle=False, **common)
    test_gen = plain_test.flow_from_directory(os.path.join(DATA_DIR, "test"), shuffle=False, **common)
    return train_gen, val_gen, test_gen


def class_weights(train_gen):
    counts = np.bincount(train_gen.classes, minlength=train_gen.num_classes).astype(float)
    w = (counts.sum() / (len(counts) * counts)) ** 0.5  # sqrt-dampened "balanced"
    return {i: float(v) for i, v in enumerate(w)}


def evaluate(model, test_gen, names):
    probs = model.predict(test_gen, verbose=0)
    y_pred, y_true = probs.argmax(1), test_gen.classes
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    print(f"\nTEST accuracy: {acc:.4f}\n{'class':<10}{'prec':>7}{'recall':>8}{'support':>9}")
    per_class = {}
    for i, name in enumerate(names):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        per_class[name] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                           "support": int(cm[i].sum())}
        print(f"{name:<10}{prec:>7.2f}{rec:>8.2f}{cm[i].sum():>9}")
    return acc, cm, per_class


def save_plots(history, cm, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed - skipping plots (pip install matplotlib).")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(MODELS_DIR, "training_curves.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Normalised confusion matrix (test)")
    fig.tight_layout(); fig.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print(f"[ERROR] Dataset not found at {DATA_DIR}/train")
        print("Download FER-2013 from https://www.kaggle.com/datasets/msambare/fer2013")
        return

    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_gen, val_gen, test_gen = get_generators(args.batch_size)

    # Save label order EXACTLY as Keras indexed it - inference reads this file.
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(train_gen.class_indices, f)
    names = [n.capitalize() for n, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
    print("Class order:", names)

    model = build_emotion_cnn(num_classes=train_gen.num_classes)
    model.compile(optimizer=Adam(args.lr),
                  loss=CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
    model.summary()

    cbs = [
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs,
                        class_weight=class_weights(train_gen), callbacks=cbs)

    acc, cm, per_class = evaluate(model, test_gen, names)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": acc, "best_val_accuracy": float(max(history.history["val_accuracy"])),
                   "per_class": per_class, "params": int(model.count_params())}, f, indent=2)
    save_plots(history, cm, names)
    print(f"\nModel saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
