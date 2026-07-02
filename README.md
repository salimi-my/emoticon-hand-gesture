# Emoticon Hand Gesture Classifier

A hand gesture classification system that recognises 5 classes using **MediaPipe hand cropping** + **MobileNetV2 transfer learning**:

| Class | Gesture | Emoji |
|-------|---------|-------|
| peace | Peace / Victory sign | ✌️ |
| okay | OK sign | 👌 |
| fist | Closed fist | 👊 |
| thumbs_up | Thumbs up | 👍 |
| high_five | Open palm | 🤚 |

Built for **CSC583 Group Project** using TensorFlow/Keras, MediaPipe, OpenCV, and Gradio.

---

## Project Structure

```
emoticon-hand-gesture/
├── dataset/
│   ├── train/          # Training images (not in repo — download from Google Drive)
│   └── test/           # Test images (≥15 per class)
├── model/
│   ├── hand_gesture_model.keras   # Pre-trained model (included in repo)
│   ├── class_indices.npy
│   └── training_curves.png
├── src/
│   ├── config.py           # Shared settings
│   ├── hand_crop.py        # MediaPipe hand detection & crop
│   ├── predict.py          # Shared inference logic
│   ├── preprocess_dataset.py  # Re-crop existing images
│   ├── train.py            # MobileNetV2 training
│   ├── evaluate.py         # Metrics & confusion matrix
│   └── app.py              # Gradio web app
├── collect_images.py   # Webcam dataset collection (hand-cropped)
└── requirements.txt
```

---

## Setup

> **Python version note:** TensorFlow requires Python 3.11. Download it from [python.org](https://www.python.org/downloads/release/python-3119/) if you don't have it.

**1. Create the virtual environment**

```bash
python3.11 -m venv .venv
```

> On Windows, if `python3.11` is not recognised, try `py -3.11 -m venv .venv`

**2. Activate the virtual environment**

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the MediaPipe hand model** (required once)

```bash
curl -L -o model/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

> Make sure the virtual environment is active before running any scripts below.

A **pre-trained model** is included in the repo (`model/hand_gesture_model.keras`), so you can skip training and go straight to [Step 4 — Run the Web App](#step-4--run-the-web-app). Download the training images (step 5) only if you want to retrain or evaluate from scratch.

**5. Download the training images** (optional — only needed to retrain or evaluate)

Training images are not stored in this repository (too large for Git). Download them from Google Drive:

**[train-data.zip on Google Drive](https://drive.google.com/file/d/19h37yYgpA5GhXJsR4IJSqL2g-DGhIdbW/view?usp=sharing)**

Extract the zip so images land under `dataset/train/` (one folder per class: `peace`, `okay`, `fist`, `thumbs_up`, `high_five`):

```bash
unzip train-data.zip -d dataset/train
```

> If the zip already contains a `train/` folder, extract to `dataset/` instead: `unzip train-data.zip -d dataset`

Verify you have images in each class folder before running `train.py` or `evaluate.py`. To collect your own images instead of using the archive, follow Step 1 below.

---

## Step 1 — Collect Your Dataset (optional)

Use the webcam collection tool if you want to build your own dataset instead of using the Google Drive archive. Images are **automatically hand-cropped** with MediaPipe before saving (no UI overlays saved):

```bash
python collect_images.py
```

**Controls:**
- Press `1`–`5` to select the gesture class (1=peace, 2=okay, 3=fist, 4=thumbs_up, 5=high_five)
- Press `SPACE` to capture and save the hand-cropped frame
- Press `t` to toggle between train / test split
- Press `q` to quit

Aim for at least **35 train** and **15 test** images per class (250 total minimum).

**Tips for a good dataset:**
- Vary backgrounds, lighting, and hand orientations
- Include multiple group members' hands
- Capture at different distances from the camera
- Keep your hand inside the green bounding box when it appears

---

## Step 1b — Reprocess Existing Images (optional)

If you collected images before hand-cropping was added, run this to **hand-crop** them in place.
This does **not** train the model — it only updates image files under `dataset/`.

```bash
python src/preprocess_dataset.py
```

Shows a progress bar with **done / left / ETA**. Useful options:

```bash
python src/preprocess_dataset.py --dir dataset/train   # crop dataset/train/ only
python src/preprocess_dataset.py --dir dataset/test    # crop dataset/test/ only
python src/preprocess_dataset.py --limit 200              # max 200 images per class
```

To **train the model**, use `python src/train.py` (separate step).

This re-crops all images in `dataset/train/` and `dataset/test/` using MediaPipe.

---

## Step 2 — Train the Model (optional)

Skip this step if you are using the pre-trained model already in `model/`. Training on the full dataset can take a long time.

```bash
python src/train.py
```

This will:
- Load and augment hand-cropped images from `dataset/train/`
- Train a **MobileNetV2 transfer learning** model (frozen ImageNet base)
- Apply **class weights** to balance uneven class counts
- Save the best model to `model/hand_gesture_model.keras`
- Display training/validation accuracy and loss curves

---

## Step 3 — Evaluate the Model (optional)

Requires test images in `dataset/test/`. Regenerates `confusion_matrix.png` and `metrics_report.txt` (not stored in the repo).

```bash
python src/evaluate.py
```

Outputs:
- Per-class accuracy, precision, recall, F1-score
- Weighted average metrics
- Confusion matrix heatmap
- Saves plots to `model/`

---

## Step 4 — Run the Web App

```bash
python src/app.py
```

Opens a Gradio web interface at `http://localhost:7860` with:
- **Upload tab** — upload an image, auto-classifies instantly
- **Webcam tab** — live streaming prediction (MediaPipe crop + classify every frame)

---

## Model Architecture

```
Webcam/Image
  → MediaPipe hand detection & crop
  → Resize to 128×128
  → MobileNetV2 (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(128) + Dropout(0.4)
  → Dense(5) + Softmax
```

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Augmentation:** Rotation, flip, zoom, shift, brightness
- **Class weights:** Balanced automatically during training

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| TensorFlow/Keras | MobileNetV2 transfer learning |
| MediaPipe | Hand detection and cropping |
| OpenCV | Webcam capture, image I/O |
| Gradio | Web-based GUI (upload + live webcam) |
| NumPy | Array operations |
| Matplotlib | Training curves, visualisations |
| Seaborn | Confusion matrix heatmap |
| scikit-learn | Metrics and class weights |
| Pillow | Image loading in the web app |
