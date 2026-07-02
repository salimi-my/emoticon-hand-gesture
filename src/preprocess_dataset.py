"""
Batch-preprocess existing dataset images with MediaPipe hand cropping.

Run this if you already collected images before hand-crop was added,
or want to re-crop all images in dataset/train and dataset/test.

Usage:
  python src/preprocess_dataset.py
  python src/preprocess_dataset.py --dir dataset/train
  python src/preprocess_dataset.py --limit 200
"""

import argparse
import os
import sys

import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import DATASET_DIR, IMAGE_EXTENSIONS, TRAIN_DIR, TEST_DIR  # noqa: E402
from hand_crop import crop_hand_bgr  # noqa: E402


def collect_image_paths(folder: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def preprocess_paths(paths: list[str], label: str) -> tuple[int, int, int]:
    """Returns (processed, hand_detected, fallback)."""
    processed = hand_detected = fallback = 0
    total = len(paths)

    if total == 0:
        print(f"  No images found in {label}.")
        return 0, 0, 0

    print(f"  Found {total:,} images in {label}")

    with tqdm(paths, desc=f"  {label}", unit="img", ncols=80) as bar:
        for path in bar:
            frame = cv2.imread(path)
            if frame is None:
                bar.write(f"  Skipped (unreadable): {path}")
                continue

            cropped_rgb, detected = crop_hand_bgr(frame)
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, cropped_bgr)

            processed += 1
            if detected:
                hand_detected += 1
            else:
                fallback += 1

            left = total - processed
            bar.set_postfix(done=processed, left=left, hand=hand_detected)

    return processed, hand_detected, fallback


def main():
    parser = argparse.ArgumentParser(
        description="Hand-crop images in dataset/ folders with MediaPipe (does NOT train the model)",
    )
    parser.add_argument(
        "--dir",
        choices=["dataset/train", "dataset/test", "both"],
        default="both",
        help=(
            "Which dataset directory to hand-crop (default: both). "
            "This is NOT model training — use src/train.py for that."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per class folder to process (useful for quick tests)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset Preprocessor — MediaPipe Hand Crop")
    print("=" * 60)

    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Dataset folder not found: {DATASET_DIR}")
        sys.exit(1)

    folders = []
    if args.dir in ("dataset/train", "both"):
        folders.append(("dataset/train", TRAIN_DIR))
    if args.dir in ("dataset/test", "both"):
        folders.append(("dataset/test", TEST_DIR))

    total_processed = total_hand = total_fallback = 0

    for folder_label, folder_path in folders:
        if not os.path.isdir(folder_path):
            print(f"\nSkipping missing folder: {folder_path}")
            continue

        print(f"\nHand-cropping {folder_label}...")

        if args.limit:
            paths = []
            for class_dir in sorted(os.listdir(folder_path)):
                class_path = os.path.join(folder_path, class_dir)
                if not os.path.isdir(class_path) or class_dir.startswith("."):
                    continue
                class_paths = collect_image_paths(class_path)[: args.limit]
                paths.extend(class_paths)
        else:
            paths = collect_image_paths(folder_path)

        processed, hand_detected, fallback = preprocess_paths(paths, folder_label)
        total_processed += processed
        total_hand += hand_detected
        total_fallback += fallback
        print(
            f"  Finished {folder_label}: {processed:,} done — "
            f"{hand_detected:,} hand crop, {fallback:,} center fallback"
        )

    print("\n" + "=" * 60)
    print(f"Done. {total_processed:,} images processed.")
    print(f"  Hand detected  : {total_hand:,}")
    print(f"  Center fallback: {total_fallback:,}")
    print("=" * 60)
    print("\nNext step: python src/train.py")


if __name__ == "__main__":
    main()
