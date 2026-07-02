"""Shared configuration for the Hand Gesture Classifier."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "hand_gesture_model.keras")
MODEL_PATH_LEGACY = os.path.join(MODEL_DIR, "hand_gesture_model.h5")
CLASS_IDX_PATH = os.path.join(MODEL_DIR, "class_indices.npy")
CURVES_PATH = os.path.join(MODEL_DIR, "training_curves.png")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_report.txt")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5
VALIDATION_SPLIT = 0.2
SEED = 42

CLASS_NAMES = ["fist", "high_five", "okay", "peace", "thumbs_up"]

CLASS_EMOJIS = {
    "peace": "✌️  Peace",
    "okay": "👌  Okay",
    "fist": "👊  Fist",
    "thumbs_up": "👍  Thumbs Up",
    "high_five": "🤚  High Five",
}

TARGET_COUNTS = {"train": 35, "test": 15}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
