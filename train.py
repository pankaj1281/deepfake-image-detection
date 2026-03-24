"""
train.py — Full training pipeline for the Fake Image Detection system.

Usage:
    python train.py [--dataset_dir DATASET_DIR] [--model_type {cnn,hybrid}]
                    [--image_size IMAGE_SIZE] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--output_dir OUTPUT_DIR]

Example:
    python train.py --dataset_dir dataset --model_type hybrid --epochs 20
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import tensorflow as tf

from utils.data_loader import DataLoader
from models.cnn_model import build_cnn_model
from models.hybrid_model import build_hybrid_model


# ─────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fake image detection model."
    )
    parser.add_argument(
        "--dataset_dir", default="dataset",
        help="Root dataset directory containing real/ and fake/ sub-folders."
    )
    parser.add_argument(
        "--model_type", default="hybrid", choices=["cnn", "hybrid"],
        help="Model architecture to train (default: hybrid)."
    )
    parser.add_argument(
        "--image_size", type=int, default=128,
        help="Resize images to IMAGE_SIZE × IMAGE_SIZE (default: 128)."
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Maximum training epochs (default: 20)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)."
    )
    parser.add_argument(
        "--output_dir", default="models",
        help="Directory to save the trained model and plots (default: models)."
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable training-time data augmentation."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_training_history(history, output_dir: str) -> None:
    """Plot and save training / validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training history plot saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, output_dir: str) -> None:
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    End-to-end training pipeline.

    Steps:
    1. Load and preprocess dataset.
    2. Build model (CNN or Hybrid).
    3. Train with EarlyStopping + ModelCheckpoint callbacks.
    4. Evaluate on validation set.
    5. Save model, plots, and evaluation report.

    Args:
        args: Parsed command-line arguments.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    image_size = (args.image_size, args.image_size)

    # ── 1. Load data ─────────────────────────
    print("[INFO] Loading dataset …")
    loader = DataLoader(
        dataset_dir=args.dataset_dir,
        image_size=image_size,
    )
    loader.load()

    train_ds, val_ds = loader.get_tf_datasets(
        batch_size=args.batch_size,
        augment_train=not args.no_augment,
    )

    # ── 2. Build model ───────────────────────
    print(f"[INFO] Building {args.model_type} model …")
    if args.model_type == "hybrid":
        model = build_hybrid_model(
            spatial_input_shape=(*image_size, 3),
            fft_input_shape=(*image_size, 1),
        )
    else:
        model = build_cnn_model(input_shape=(*image_size, 3))

    model.summary()

    # ── 3. Callbacks ─────────────────────────
    best_model_path = os.path.join(
        args.output_dir, f"best_{args.model_type}_model.keras"
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── 4. Train ─────────────────────────────
    print(f"[INFO] Training for up to {args.epochs} epochs …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── 5. Evaluate ──────────────────────────
    print("[INFO] Evaluating on validation set …")
    if args.model_type == "hybrid":
        x_val = {
            "spatial_input": loader.x_val_spatial,
            "fft_input": loader.x_val_fft,
        }
    else:
        x_val = loader.x_val_spatial

    y_prob = model.predict(x_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = loader.y_val

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'─'*50}")
    print(f"  Validation Accuracy : {acc:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # ── 6. Save artefacts ────────────────────
    plot_training_history(history, args.output_dir)
    plot_confusion_matrix(y_true, y_pred, args.output_dir)

    # Save final model (may differ from best checkpoint if overfitted)
    final_model_path = os.path.join(
        args.output_dir, f"{args.model_type}_model_final.keras"
    )
    model.save(final_model_path)
    print(f"[INFO] Final model saved → {final_model_path}")
    print(f"[INFO] Best model saved  → {best_model_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
