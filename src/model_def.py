"""
model_def.py
------------
Defines the CNN architecture used for facial emotion classification.

Kept in its own file (separate from training/inference logic) so it's easy
to inspect, modify, and explain layer-by-layer in an interview setting.

Input : 48x48 grayscale face image (1 channel)
Output: probability distribution over 7 emotion classes
"""

from tensorflow.keras import layers, models

EMOTION_CLASSES = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

IMG_SIZE = 48


def build_emotion_cnn(num_classes: int = len(EMOTION_CLASSES)) -> models.Sequential:
    """
    Builds a compact CNN for 48x48 grayscale emotion classification.

    Design rationale:
    - 3 convolutional blocks progressively extract low -> high level facial features
      (edges/gradients -> facial parts like eyes/mouth curvature -> holistic expression)
    - BatchNorm after each conv layer stabilizes training and speeds convergence
    - Dropout layers combat overfitting, since FER-2013 is a relatively small/noisy dataset
    - GlobalAveragePooling instead of Flatten+huge Dense layer keeps parameter count low
      (important if you later want to quantize/deploy this on constrained hardware)
    """
    model = models.Sequential(name="emotion_cnn")

    # Block 1
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Classifier head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print the architecture summary.
    m = build_emotion_cnn()
    m.summary()
