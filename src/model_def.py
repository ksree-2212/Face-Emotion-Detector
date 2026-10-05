"""
model_def.py
------------
CNN architecture + label handling for facial emotion classification.

Input : 48x48 grayscale face (1 channel)
Output: probability distribution over 7 emotion classes

IMPORTANT (label order): Keras `flow_from_directory` assigns class indices in
ALPHABETICAL folder order. The labels below MUST follow that same order, and
train_model.py also saves the exact mapping to models/class_names.json, which
the detection scripts load, so training and inference can never disagree.
"""
import json
import os

IMG_SIZE = 48
# Alphabetical == folder order of data/fer2013/train
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")


def load_class_names():
    """Labels in the exact index order used during training."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            mapping = json.load(f)  # {"angry": 0, ...}
        return [n.capitalize() for n, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    return EMOTION_CLASSES


def build_emotion_cnn(num_classes: int = len(EMOTION_CLASSES)):
    """
    Compact CNN trained FROM SCRATCH (random init, no pretrained weights).

    - 3 conv blocks: edges -> facial parts -> expression-level features
    - BatchNorm stabilises training; Dropout fights overfitting (FER-2013 is small/noisy)
    - GlobalAveragePooling keeps the parameter count low (good for edge/TFLite later)
    """
    from tensorflow.keras import layers, models

    m = models.Sequential(name="emotion_cnn")
    m.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    for filters, n_convs, drop in [(32, 2, 0.25), (64, 2, 0.25), (128, 1, 0.30)]:
        for _ in range(n_convs):
            m.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
            m.add(layers.BatchNormalization())
        m.add(layers.MaxPooling2D((2, 2)))
        m.add(layers.Dropout(drop))

    m.add(layers.GlobalAveragePooling2D())
    m.add(layers.Dense(128, activation="relu"))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(num_classes, activation="softmax"))
    return m


if __name__ == "__main__":
    build_emotion_cnn().summary()
