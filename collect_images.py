"""
Webcam dataset collection tool for the Hand Gesture Classifier.

Uses MediaPipe to crop the hand region before saving, so training images
focus on the gesture rather than background.

Controls:
  1-5     : Select the gesture class (shown on-screen)
  SPACE   : Capture and save the current frame (hand-cropped)
  t       : Toggle split between 'train' and 'test'
  q       : Quit

Usage:
  python collect_images.py
"""

import cv2
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import DATASET_DIR, TARGET_COUNTS  # noqa: E402
from hand_crop import crop_hand_bgr, draw_hand_box  # noqa: E402

CLASSES = {
    "1": "peace",
    "2": "okay",
    "3": "fist",
    "4": "thumbs_up",
    "5": "high_five",
}

CLASS_LABELS = {
    "peace": "Peace (V)",
    "okay": "Okay (OK)",
    "fist": "Fist",
    "thumbs_up": "Thumbs Up",
    "high_five": "High Five",
}


def count_images(class_name: str, split: str) -> int:
    folder = os.path.join(DATASET_DIR, split, class_name)
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def next_filename(class_name: str, split: str) -> str:
    folder = os.path.join(DATASET_DIR, split, class_name)
    os.makedirs(folder, exist_ok=True)
    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    indices = []
    for f in existing:
        name = os.path.splitext(f)[0]
        if name.startswith(f"{class_name}_"):
            try:
                indices.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    next_idx = max(indices, default=0) + 1
    return os.path.join(folder, f"{class_name}_{next_idx:04d}.jpg")


def draw_overlay(frame, current_class: str, split: str, last_saved: str, flash: bool, hand_detected: bool):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    label = CLASS_LABELS.get(current_class, current_class)
    cv2.putText(frame, f"Class: {label}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)

    split_color = (0, 200, 255) if split == "train" else (255, 100, 0)
    cv2.putText(frame, f"Split: {split.upper()}  [T] to toggle", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, split_color, 2)

    hand_status = "Hand OK" if hand_detected else "No hand (center crop)"
    hand_color = (0, 255, 150) if hand_detected else (0, 165, 255)
    cv2.putText(frame, hand_status, (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_color, 1)

    counts_text = "  ".join(
        f"{k[0].upper()}:{count_images(v, split)}/{TARGET_COUNTS[split]}"
        for k, v in CLASSES.items()
    )
    cv2.putText(frame, counts_text, (w - 420, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "[SPACE] Capture   [1-5] Class   [Q] Quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if flash:
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, 0), (w, h), (255, 255, 255), -1)
        cv2.addWeighted(flash_overlay, 0.4, frame, 0.6, 0, frame)
        if last_saved:
            cv2.putText(frame, f"Saved: {os.path.basename(last_saved)}", (15, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def print_summary():
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    for split in ["train", "test"]:
        print(f"\n  {split.upper()} (target: {TARGET_COUNTS[split]} per class)")
        for class_name in CLASSES.values():
            count = count_images(class_name, split)
            target = TARGET_COUNTS[split]
            bar = "#" * count + "-" * max(0, target - count)
            status = "OK" if count >= target else f"{target - count} more needed"
            print(f"    {class_name:<12} [{bar}] {count}/{target}  {status}")
    print()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Connect a camera and try again.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_class = "peace"
    split = "train"
    last_saved = ""
    flash_until = 0.0
    last_hand_detected = False

    print("\nHand Gesture Dataset Collector (MediaPipe hand crop)")
    print("-----------------------------------------------------")
    print("Keys: [SPACE] capture  [1-5] select class  [T] toggle split  [Q] quit")
    print_summary()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        raw_frame = cv2.flip(frame, 1)
        display_frame = raw_frame.copy()
        display_frame = draw_hand_box(display_frame)

        # Check hand detection on raw frame for status indicator
        _, last_hand_detected = crop_hand_bgr(raw_frame)

        flash = time.time() < flash_until
        draw_overlay(display_frame, current_class, split, last_saved, flash, last_hand_detected)
        cv2.imshow("Hand Gesture Collector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif chr(key) in CLASSES:
            current_class = CLASSES[chr(key)]
            print(f"  Selected class: {current_class}")
        elif key == ord("t"):
            split = "test" if split == "train" else "train"
            print(f"  Switched to split: {split}")
        elif key == ord(" "):
            cropped_rgb, hand_detected = crop_hand_bgr(raw_frame)
            path = next_filename(current_class, split)
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, cropped_bgr)
            last_saved = path
            flash_until = time.time() + 0.2
            count = count_images(current_class, split)
            target = TARGET_COUNTS[split]
            status = "hand crop" if hand_detected else "center crop fallback"
            print(f"  [{split}/{current_class}] Saved {os.path.basename(path)} ({status})  ({count}/{target})")
            if count >= target:
                print(f"  Target reached for [{split}/{current_class}]!")

    cap.release()
    cv2.destroyAllWindows()
    print_summary()


if __name__ == "__main__":
    main()
