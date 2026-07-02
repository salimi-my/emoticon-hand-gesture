"""
Hand detection and cropping using MediaPipe Tasks API.

Crops the detected hand region from an image so the classifier focuses on
gesture shape rather than background clutter.
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

_image_landmarker = None
_video_landmarker = None
_video_timestamp_ms = 0


def _ensure_model_exists():
    if not os.path.isfile(LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe model not found: {LANDMARKER_MODEL_PATH}\n"
            "Download it with:\n"
            "curl -L -o model/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )


def _get_image_landmarker():
    global _image_landmarker
    if _image_landmarker is None:
        _ensure_model_exists()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _image_landmarker = vision.HandLandmarker.create_from_options(options)
    return _image_landmarker


def _get_video_landmarker():
    global _video_landmarker
    if _video_landmarker is None:
        _ensure_model_exists()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _video_landmarker = vision.HandLandmarker.create_from_options(options)
    return _video_landmarker


def _landmarks_to_bbox(hand_landmarks, width: int, height: int, padding: float = 0.25):
    xs = [lm.x * width for lm in hand_landmarks]
    ys = [lm.y * height for lm in hand_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * padding
    pad_y = box_h * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(width, int(x_max + pad_x))
    y2 = min(height, int(y_max + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _center_square_crop(image: np.ndarray) -> np.ndarray:
    """Fallback when no hand is detected — crop central square region."""
    h, w = image.shape[:2]
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    return image[y1 : y1 + size, x1 : x1 + size]


def _detect_landmarks(image_rgb: np.ndarray, streaming: bool = False):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))

    if streaming:
        global _video_timestamp_ms
        _video_timestamp_ms += 33
        landmarker = _get_video_landmarker()
        result = landmarker.detect_for_video(mp_image, _video_timestamp_ms)
    else:
        landmarker = _get_image_landmarker()
        result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


def crop_hand_from_array(
    image_rgb: np.ndarray,
    padding: float = 0.25,
    fallback_center: bool = True,
    streaming: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Crop hand region from an RGB numpy array.

    Returns:
        (cropped_rgb, hand_detected)
    """
    h, w = image_rgb.shape[:2]
    landmarks = _detect_landmarks(image_rgb, streaming=streaming)

    if landmarks:
        bbox = _landmarks_to_bbox(landmarks, w, h, padding)
        if bbox:
            x1, y1, x2, y2 = bbox
            return image_rgb[y1:y2, x1:x2], True

    if fallback_center:
        return _center_square_crop(image_rgb), False
    return image_rgb, False


def crop_hand_pil(
    pil_image: Image.Image,
    padding: float = 0.25,
    fallback_center: bool = True,
    streaming: bool = False,
) -> tuple[Image.Image, bool]:
    """Crop hand from a PIL image. Returns (cropped_pil, hand_detected)."""
    rgb = np.array(pil_image.convert("RGB"))
    cropped, detected = crop_hand_from_array(
        rgb, padding, fallback_center, streaming=streaming,
    )
    return Image.fromarray(cropped), detected


def crop_hand_bgr(
    frame_bgr: np.ndarray,
    padding: float = 0.25,
    fallback_center: bool = True,
    streaming: bool = False,
) -> tuple[np.ndarray, bool]:
    """Crop hand from an OpenCV BGR frame. Returns (cropped_rgb, hand_detected)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return crop_hand_from_array(rgb, padding, fallback_center, streaming=streaming)


def draw_hand_box(frame_bgr: np.ndarray, padding: float = 0.25) -> np.ndarray:
    """Draw bounding box around detected hand (for collection preview)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    landmarks = _detect_landmarks(rgb, streaming=False)

    if landmarks:
        bbox = _landmarks_to_bbox(landmarks, w, h, padding)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 150), 2)
            cv2.putText(
                frame_bgr, "Hand detected", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1,
            )
    else:
        cv2.putText(
            frame_bgr, "No hand — using center crop", (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1,
        )
    return frame_bgr
