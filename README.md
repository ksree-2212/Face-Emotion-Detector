# Real-Time Face & Emotion Detection System

Detects faces from a webcam/image and classifies the emotion (Angry, Disgust, Fear,
Happy, Neutral, Sad, Surprise) with a CNN **trained from scratch on FER-2013**.

## Is anything pretrained?
| Component | Pretrained? | Details |
|---|---|---|
| Emotion classifier (`src/model_def.py`) | **No** | Custom CNN, random initialisation, trained by `train_model.py` on FER-2013. No transfer learning, no ImageNet/VGG/ResNet weights. |
| Face localiser (Haar Cascade) | Yes (classical) | `haarcascade_frontalface_default.xml` ships with OpenCV. It only finds *where* the face is; it does not predict emotion. |

## Pipeline
Frame -> grayscale -> Haar face detection -> crop, resize 48x48, /255 -> CNN -> softmax ->
label + confidence (below 45% shown as "Uncertain"); live mode averages the last 8 frames per face.

## Setup & run
```bash
python -m venv .venv          # Python 3.10 - 3.12 recommended
.venv\Scripts\activate        # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Put FER-2013 in data/fer2013/train/<class>/ and data/fer2013/test/<class>/
# 2. Train (outputs: models/emotion_cnn.h5, class_names.json, metrics.json,
#                    training_curves.png, confusion_matrix.png)
python src/train_model.py --epochs 60

# 3. Use it
python src/detect_live.py
python src/detect_image.py --image photo.jpg --output result.jpg
```

## Training methodology (for the report)
- Dataset: FER-2013, 28,709 train / 7,178 test, 48x48 grayscale, 7 classes (imbalanced: disgust = 436 images).
- Split: 90% train / 10% validation (checkpointing + early stopping); the official test set is used **once**, for the final report.
- Augmentation: rotation, shift, zoom, brightness, horizontal flip (train only).
- Imbalance: square-root-dampened class weights. Loss: categorical cross-entropy with label smoothing 0.05.
- Optimiser: Adam 1e-3 with ReduceLROnPlateau; EarlyStopping (patience 10) on validation accuracy.
- Evaluation: test accuracy, per-class precision/recall, confusion matrix (all auto-saved).

## Known limitations (good for the discussion section)
- FER-2013 labels are noisy (human agreement is only ~65%), so ~60-68% test accuracy is typical for a small from-scratch CNN.
- Disgust/Fear are the weakest classes (few, ambiguous samples); Happy/Surprise are strongest.
- Haar cascades miss tilted/occluded faces; swap in OpenCV's DNN detector for better recall at lower FPS.
