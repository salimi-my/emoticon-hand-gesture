"""
CNN training script for the Hand Gesture Classifier.

Uses transfer learning with MobileNetV2 (ImageNet weights, frozen base)
and a custom classifier head for 5 gesture classes.

Architecture:
  Input (128x128x3)
  -> MobileNetV2 (frozen, feature extractor)
  -> GlobalAveragePooling2D
  -> Dense(128) + Dropout
  -> Dense(5, softmax)

Usage:
  python src/train.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from config import (
    BATCH_SIZE,
    CLASS_IDX_PATH,
    CURVES_PATH,
    EPOCHS,
    IMG_SIZE,
    MODEL_DIR,
    MODEL_PATH,
    NUM_CLASSES,
    SEED,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def build_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


def compute_class_weights(train_gen) -> dict:
    """Balance learning when classes have unequal image counts."""
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return {i: float(w) for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# Model definition — MobileNetV2 transfer learning
# ---------------------------------------------------------------------------
def build_model() -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="hand_gesture_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------
def build_callbacks() -> list:
    return [
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", marker="o", markersize=3)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", marker="o", markersize=3)
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", marker="o", markersize=3)
    axes[1].plot(history.history["val_loss"], label="Val Loss", marker="o", markersize=3)
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Hand Gesture Classifier — MobileNetV2 Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
    print(f"Training curves saved to: {CURVES_PATH}")
    plt.close()


# ---------------------------------------------------------------------------
# Validation check
# ---------------------------------------------------------------------------
def validate_dataset():
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training directory not found: {TRAIN_DIR}")
        sys.exit(1)

    found_classes = [
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d)) and not d.startswith(".")
    ]
    if not found_classes:
        print("ERROR: No class folders found in dataset/train/.")
        print("Run collect_images.py first to gather your dataset.")
        sys.exit(1)

    total = 0
    print("\nDataset check:")
    for cls in sorted(found_classes):
        folder = os.path.join(TRAIN_DIR, cls)
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {cls:<14} {len(imgs)} images")
        total += len(imgs)
    print(f"  {'TOTAL':<14} {total} images\n")

    if total == 0:
        print("ERROR: Dataset is empty. Collect images first.")
        sys.exit(1)

    if total < NUM_CLASSES * 10:
        print("WARNING: Very few images. Aim for at least 35 train images per class.")
        print("Tip: run python src/preprocess_dataset.py to hand-crop existing images.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Hand Gesture Classifier — MobileNetV2 Transfer Learning")
    print("=" * 60)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Image size         : {IMG_SIZE}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Max epochs         : {EPOCHS}")
    print(f"Validation split   : {VALIDATION_SPLIT * 100:.0f}%")

    validate_dataset()

    print("Building data generators...")
    train_gen, val_gen = build_generators()

    class_indices = train_gen.class_indices
    print(f"\nClass mapping: {class_indices}")
    np.save(CLASS_IDX_PATH, class_indices)

    class_weights = compute_class_weights(train_gen)
    print(f"Class weights: {class_weights}")

    print("\nBuilding MobileNetV2 model (frozen base)...")
    model = build_model()
    model.summary()

    print("\nStarting training...\n")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=build_callbacks(),
    )

    print("\nTraining complete.")
    print(f"Best model saved to: {MODEL_PATH}")

    val_gen.reset()
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Final validation accuracy : {val_acc * 100:.2f}%")
    print(f"Final validation loss     : {val_loss:.4f}")

    plot_training_curves(history)
    print("\nNext step: run python src/evaluate.py")


if __name__ == "__main__":
    main()
