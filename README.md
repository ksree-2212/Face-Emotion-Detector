# Real-Time Face & Emotion Detection System

A real-time computer vision pipeline that detects faces from a webcam/video feed
and classifies the emotion expressed (Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust)
using a Convolutional Neural Network.

## Why this project (for Qualcomm)
Qualcomm's Snapdragon platforms power camera ISPs and on-device vision AI (auto-focus,
scene detection, portrait mode, AR/VR). This project mirrors that exact pipeline:
**capture → face localization → real-time classification**, and it's a natural way to
talk about latency, model size, and accuracy tradeoffs in an interview.

## Architecture

```
Webcam Frame
     │
     ▼
┌─────────────────────┐
│ Face Detection       │  Haar Cascade (fast, CPU-only, classical CV)
│ (OpenCV)              │  OR DNN-based detector (more accurate, heavier)
└─────────┬────────────┘
          │ cropped face ROI (48x48 grayscale)
          ▼
┌─────────────────────┐
│ Emotion Classifier    │  CNN: Conv → Pool → Conv → Pool → Dense → Softmax
│ (TensorFlow/Keras)    │  Trained on FER-2013 dataset
└─────────┬────────────┘
          │
          ▼
   Emotion label + confidence
   overlaid on video frame
```

## Files
- `src/train_model.py` — builds and trains the CNN on FER-2013 (or any labeled face-emotion dataset)
- `src/model_def.py` — the CNN architecture, isolated so it's easy to explain/modify
- `src/detect_live.py` — real-time webcam pipeline: face detection + emotion inference + overlay
- `src/detect_image.py` — same pipeline but on a single static image (useful for testing without a webcam)
- `models/` — where the trained `.h5` model gets saved
- `data/` — place the dataset here (see Dataset section)

## Dataset
This project is built to train on **FER-2013** (35,887 labeled 48x48 grayscale face images,
7 emotion classes), a standard benchmark dataset. Since it requires a Kaggle account to
download, do the following on your machine:

1. Download from: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract into `data/fer2013/train` and `data/fer2013/test` (already split by class folder)
3. Run `python src/train_model.py`

If you don't want to train from scratch, `src/detect_live.py` will still run using face
detection alone (emotion classification will just show "model not found" gracefully) —
useful to demo the pipeline structure without needing the full dataset.

## How to run

TensorFlow wheels are not available for Python 3.14, which is why `tensorflow-cpu` fails with `No matching distribution found`.
Use a supported Python version (recommended: 3.10 or 3.11) and install the standard TensorFlow package for CPU use.

```bash
# Create a virtual environment with a supported Python version
py -3.10 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy

# Step 1: Train (requires dataset, ~15-20 min on CPU for a few epochs)
python src/train_model.py

# Step 2: Run real-time detection (needs a webcam)
python src/detect_live.py

# OR test on a single image
python src/detect_image.py --image path/to/photo.jpg
```

If you already have Python 3.10/3.11 installed, replace `py -3.10` with `py -3.11` and keep the same install commands.

## Key design decisions (interview talking points)

1. **Haar Cascade vs DNN face detector**: I used Haar Cascade for the default pipeline
   because it's extremely fast and CPU-friendly (important for real-time/edge use cases,
   similar to power/latency constraints on mobile SoCs). I documented how to swap in
   OpenCV's DNN face detector (`res10_300x300_ssd`) for higher accuracy at the cost of speed —
   a classic accuracy/latency tradeoff you can discuss.

2. **48x48 grayscale input**: Keeps the model small and fast. This is a direct analogue to
   why edge AI models get quantized/shrunk — smaller input = fewer FLOPs = lower latency.

3. **Model architecture**: A compact CNN (not ResNet/VGG) intentionally — trained fast on CPU,
   and small enough to reason about every layer, which matters when an interviewer asks
   "why did you choose this architecture."

4. **Confidence thresholding**: Predictions below a confidence threshold are shown as
   "Uncertain" rather than forcing a label — a basic but important reliability decision for
   any deployed vision system.

## Possible extensions to mention in interviews
- Convert the trained model to TensorFlow Lite and benchmark size/latency (this is exactly
  what Project 2 in this series does)
- Swap Haar Cascade for a lightweight DNN detector and compare FPS
- Multi-face tracking across frames instead of independent per-frame detection
