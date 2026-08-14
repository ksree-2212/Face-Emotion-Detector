"""
train_model.py
---------------
Trains the emotion CNN on the FER-2013 dataset.

Expects data laid out as:
    data/fer2013/train/<emotion_name>/*.jpg
    data/fer2013/test/<emotion_name>/*.jpg

Usage:
    python src/train_model.py --epochs 25 --batch_size 64
"""

import argparse
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model_def import build_emotion_cnn, IMG_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fer2013")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_cnn.h5")


def get_generators(batch_size: int):
    # Data augmentation on the training set only: small rotations/shifts/zooms
    # help the model generalize since FER-2013 has limited samples per class.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, test_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print(f"[ERROR] Dataset not found at {DATA_DIR}/train")
        print("Download FER-2013 from https://www.kaggle.com/datasets/msambare/fer2013")
        print("and extract it into data/fer2013/train and data/fer2013/test")
        return

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    train_gen, test_gen = get_generators(args.batch_size)

    model = build_emotion_cnn(num_classes=train_gen.num_classes)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\nBest validation accuracy: {final_acc:.4f}")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
