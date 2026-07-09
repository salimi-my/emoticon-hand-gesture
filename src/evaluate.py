"""
Evaluation script for the Hand Gesture Classifier.

Produces:
  - Per-class accuracy, precision, recall, F1-score
  - Weighted average metrics
  - Confusion matrix heatmap  (model/confusion_matrix.png)

Usage:
  python src/evaluate.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    CM_PATH,
    IMG_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_PATH_LEGACY,
    TEST_DIR,
)
from predict import load_model_and_classes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_path():
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    if os.path.isfile(MODEL_PATH_LEGACY):
        return MODEL_PATH_LEGACY
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Train the model first: python src/train.py")
    sys.exit(1)


def validate_test_dir():
    if not os.path.isdir(TEST_DIR):
        print(f"ERROR: Test directory not found: {TEST_DIR}")
        sys.exit(1)

    classes = [
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d)) and not d.startswith(".")
    ]
    if not classes:
        print("ERROR: No class folders found in dataset/test/.")
        sys.exit(1)

    total = 0
    print("\nTest dataset:")
    for cls in sorted(classes):
        folder = os.path.join(TEST_DIR, cls)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {cls:<14} {len(imgs)} images")
        total += len(imgs)
    print(f"  {'TOTAL':<14} {total} images\n")


def build_test_generator(class_indices):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=list(class_indices.keys()),
    )


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Confusion Matrix — Hand Gesture Classifier", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {CM_PATH}")
    plt.close()


def print_metrics_table(y_true, y_pred, class_names: list):
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )
    overall_acc = accuracy_score(y_true, y_pred)

    header = "\n" + "=" * 60
    header += "\nHand Gesture Classifier — Evaluation Results"
    header += "\n" + "=" * 60

    footer = f"\nOverall Accuracy: {overall_acc * 100:.2f}%"
    footer += "\n" + "=" * 60

    full_report = header + "\n\n" + report + footer
    print(full_report)

    with open(METRICS_PATH, "w") as f:
        f.write(full_report + "\n")
    print(f"\nMetrics report saved to: {METRICS_PATH}")

    return overall_acc


def main():
    print("=" * 60)
    print("Hand Gesture Classifier — Evaluation")
    print("=" * 60)

    validate_test_dir()

    model_path = load_model_path()
    model, idx_to_class = load_model_and_classes()
    print(f"Model loaded from: {model_path}")

    class_indices = {name: idx for idx, name in idx_to_class.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"Classes: {class_names}\n")

    test_gen = build_test_generator(class_indices)

    if test_gen.samples == 0:
        print("ERROR: No test images found. Add images to dataset/test/ folders.")
        sys.exit(1)

    print("Running predictions on test set...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    overall_acc = print_metrics_table(y_true, y_pred, class_names)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

    print("\nPer-class accuracy breakdown:")
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {cls:<14} {cls_acc * 100:.1f}%  ({mask.sum()} samples)")

    print(f"\nOverall test accuracy: {overall_acc * 100:.2f}%")
    print("\nOutputs saved to model/:")
    print("  confusion_matrix.png")
    print("  metrics_report.txt")


if __name__ == "__main__":
    main()
