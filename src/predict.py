"""Shared model loading and inference utilities."""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (
    CLASS_EMOJIS,
    CLASS_IDX_PATH,
    IMG_SIZE,
    MODEL_PATH,
    MODEL_PATH_LEGACY,
)
from hand_crop import crop_hand_pil

_predict_fn = None
PREVIEW_SIZE = (256, 256)


def load_model_and_classes():
    """Load trained model and class index mapping."""
    if os.path.isfile(MODEL_PATH):
        model_path = MODEL_PATH
    elif os.path.isfile(MODEL_PATH_LEGACY):
        model_path = MODEL_PATH_LEGACY
    else:
        raise FileNotFoundError(
            f"No model found. Train first:\n  python src/train.py\n"
            f"Expected: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(model_path)

    if os.path.isfile(CLASS_IDX_PATH):
        class_indices = np.load(CLASS_IDX_PATH, allow_pickle=True).item()
    else:
        class_indices = {
            "fist": 0,
            "high_five": 1,
            "okay": 2,
            "peace": 3,
            "thumbs_up": 4,
        }

    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def _get_predict_fn(model):
    global _predict_fn
    if _predict_fn is None:

        @tf.function
        def _run(x):
            return model(x, training=False)

        _predict_fn = _run
    return _predict_fn


def preprocess_image(pil_image: Image.Image, crop_hand: bool = True) -> np.ndarray:
    """Crop hand (optional), resize, and apply MobileNetV2 preprocessing."""
    if crop_hand:
        pil_image, _ = crop_hand_pil(pil_image)

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def crop_for_model(pil_image, streaming: bool = False) -> tuple[Image.Image, bool]:
    """Return hand-cropped PIL image and whether a hand was detected."""
    return crop_hand_pil(pil_image, streaming=streaming)


def predict_probs(model, cropped_pil: Image.Image) -> np.ndarray:
    """Run model on an already cropped PIL image. Returns raw probability vector."""
    arr = preprocess_image(cropped_pil, crop_hand=False)
    predict_fn = _get_predict_fn(model)
    return predict_fn(arr)[0].numpy()


FALLBACK_NOTICE = (
    '<div class="fallback-notice-card">'
    '<span class="fallback-notice-icon" aria-hidden="true">⚠️</span>'
    '<span class="fallback-notice-text">'
    "<strong>No hand detected</strong> — using center crop fallback. "
    "Prediction may be less accurate."
    "</span>"
    "</div>"
)


def format_confidence(probs: np.ndarray, idx_to_class: dict) -> dict:
    """Convert probability vector to emoji-keyed confidence dict for gr.Label."""
    return {
        CLASS_EMOJIS.get(idx_to_class[i], idx_to_class[i]): float(probs[i])
        for i in range(len(probs))
    }


def format_fallback_notice(hand_detected: bool) -> str:
    """Return a UI notice when MediaPipe falls back to center crop."""
    return "" if hand_detected else FALLBACK_NOTICE


def crop_preview(cropped_pil: Image.Image) -> Image.Image:
    """Resize cropped hand for UI preview."""
    return cropped_pil.convert("RGB").resize(PREVIEW_SIZE, Image.Resampling.LANCZOS)


def smooth_probs(prob_history: list, window: int = 5) -> np.ndarray | None:
    """Average probabilities over the last N frames."""
    if not prob_history:
        return None
    recent = prob_history[-window:]
    return np.mean(recent, axis=0)


def predict_gesture(pil_image, model, idx_to_class, crop_hand: bool = True, streaming: bool = False):
    """
    Run inference on a PIL image.

    Returns:
        (fallback_notice, confidence_dict, hand_detected)
    """
    if pil_image is None:
        return "", {}, False

    if crop_hand:
        cropped, hand_detected = crop_hand_pil(pil_image, streaming=streaming)
    else:
        cropped, hand_detected = pil_image, True

    probs = predict_probs(model, cropped)
    confidence = format_confidence(probs, idx_to_class)
    return format_fallback_notice(hand_detected), confidence, hand_detected
