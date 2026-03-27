"""
train.py — Full training pipeline for the Fake Image Detection system.

Usage:
    python train.py [--dataset_dir DATASET_DIR] [--model_type {cnn,hybrid,efficientnet}]
                    [--image_size IMAGE_SIZE] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--output_dir OUTPUT_DIR]
                    [--fine_tune] [--fine_tune_epochs FINE_TUNE_EPOCHS]

Example:
    python train.py --dataset_dir dataset --model_type efficientnet --epochs 20 --fine_tune
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
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from utils.data_loader import DataLoader
from models.cnn_model import build_cnn_model
from models.hybrid_model import build_hybrid_model
from models.efficientnet_model import build_efficientnet_hybrid_model


# ─────────────────────────────────────────────
# GPU / mixed-precision setup
# ─────────────────────────────────────────────

def _configure_gpu() -> None:
    """Enable GPU memory growth and mixed precision when a GPU is available.

    Mixed precision (``mixed_float16``) speeds up training on Tensor Core GPUs
    (Volta, Turing, Ampere and newer) by performing computation in float16 while
    keeping master weights in float32.  It is safe for this model, but on very
    old GPUs or with custom loss functions that accumulate small gradients you
    may occasionally see NaN losses — in that case re-run without a GPU or
    disable mixed precision by removing the ``set_global_policy`` call below.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision (float16 compute, float32 weights) roughly doubles
        # throughput on Tensor Core GPUs (Volta, Turing, Ampere, etc.)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"[INFO] GPU detected ({len(gpus)} device(s)) — mixed precision enabled.")
    else:
        print("[INFO] No GPU detected — training on CPU.")


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
        "--model_type", default="hybrid", choices=["cnn", "hybrid", "efficientnet"],
        help="Model architecture to train (default: hybrid)."
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Resize images to IMAGE_SIZE × IMAGE_SIZE (default: 224)."
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
    parser.add_argument(
        "--fine_tune", action="store_true",
        help=(
            "Enable fine-tuning phase after initial training (efficientnet only). "
            "Unfreezes the EfficientNetB0 backbone and continues training at a lower LR."
        ),
    )
    parser.add_argument(
        "--fine_tune_epochs", type=int, default=10,
        help="Number of additional fine-tuning epochs (default: 10)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for converting probability to binary label (default: 0.5)."
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

def _build_callbacks(best_model_path: str) -> list:
    """Return the standard set of training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
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


def train(args: argparse.Namespace) -> None:
    """
    End-to-end training pipeline.

    Steps:
    1. Configure GPU / mixed precision.
    2. Load and preprocess dataset.
    3. Print class distribution and compute class weights.
    4. Build model (CNN, Hybrid, or EfficientNet hybrid).
    5. Train with EarlyStopping + ModelCheckpoint callbacks.
    6. (Optional) Fine-tune the EfficientNet backbone at a lower LR.
    7. Evaluate on validation set; print predictions distribution.
    8. Save model, plots, and evaluation report.

    Args:
        args: Parsed command-line arguments.
    """
    _configure_gpu()
    os.makedirs(args.output_dir, exist_ok=True)

    # EfficientNetB0 expects at least 32×32; recommend 224×224 for best results
    if args.model_type == "efficientnet" and args.image_size < 224:
        print(
            "[WARNING] image_size < 224 with efficientnet may hurt accuracy. "
            "Consider --image_size 224 for best results."
        )

    image_size = (args.image_size, args.image_size)

    # ── 1. Load data ─────────────────────────
    print("[INFO] Loading dataset …")
    loader = DataLoader(
        dataset_dir=args.dataset_dir,
        image_size=image_size,
    )
    loader.load()

    # ── 2. Class distribution & class weights ─
    n_real = int(np.sum(loader.y_train == 0))
    n_fake = int(np.sum(loader.y_train == 1))
    n_total = len(loader.y_train)
    print(f"\n{'─'*50}")
    print("  Class distribution (training set):")
    print(f"    Real (0) : {n_real:>6}  ({n_real/n_total:.1%})")
    print(f"    Fake (1) : {n_fake:>6}  ({n_fake/n_total:.1%})")
    print(f"    Total    : {n_total:>6}")
    print(f"{'─'*50}\n")

    classes = np.unique(loader.y_train)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=loader.y_train,
    )
    class_weight_dict = dict(zip(classes.tolist(), raw_weights.tolist()))
    print(f"[INFO] Class weights: {class_weight_dict}")

    train_ds, val_ds = loader.get_tf_datasets(
        batch_size=args.batch_size,
        augment_train=not args.no_augment,
    )

    # ── 3. Build model ───────────────────────
    print(f"[INFO] Building {args.model_type} model …")
    if args.model_type == "efficientnet":
        model = build_efficientnet_hybrid_model(
            spatial_input_shape=(*image_size, 3),
            fft_input_shape=(*image_size, 1),
            freeze_backbone=True,   # Phase 1: frozen backbone
        )
    elif args.model_type == "hybrid":
        model = build_hybrid_model(
            spatial_input_shape=(*image_size, 3),
            fft_input_shape=(*image_size, 1),
        )
    else:
        model = build_cnn_model(input_shape=(*image_size, 3))

    model.summary()

    # ── 4. Callbacks ─────────────────────────
    best_model_path = os.path.join(
        args.output_dir, f"best_{args.model_type}_model.keras"
    )
    callbacks = _build_callbacks(best_model_path)

    # ── 5. Train (phase 1) ───────────────────
    print(f"[INFO] Training for up to {args.epochs} epochs …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    # ── 5b. Fine-tune phase (efficientnet only) ──
    if args.model_type == "efficientnet" and args.fine_tune:
        print(
            f"\n[INFO] Fine-tuning EfficientNetB0 backbone "
            f"for up to {args.fine_tune_epochs} additional epochs …"
        )
        # Unfreeze the backbone and re-compile with a lower LR
        backbone_layer = model.get_layer("efficientnetb0_backbone")
        backbone_layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        fine_tune_callbacks = _build_callbacks(best_model_path)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            class_weight=class_weight_dict,
        )

        # Merge history dicts safely (only merge keys present in both runs)
        for key in list(history.history.keys()):
            if key in history_ft.history:
                history.history[key] = (
                    history.history[key] + history_ft.history[key]
                )

    # ── 6. Evaluate ──────────────────────────
    print("[INFO] Evaluating on validation set …")
    if args.model_type in ("hybrid", "efficientnet"):
        x_val = {
            "spatial_input": loader.x_val_spatial,
            "fft_input": loader.x_val_fft,
        }
    else:
        x_val = loader.x_val_spatial

    y_prob = model.predict(x_val, verbose=0).flatten()
    y_pred = (y_prob >= args.threshold).astype(int)
    y_true = loader.y_val

    # Predictions distribution
    n_pred_fake = int(np.sum(y_pred == 1))
    n_pred_real = int(np.sum(y_pred == 0))
    print(f"\n{'─'*50}")
    print("  Predictions distribution (validation set):")
    print(f"    Predicted Real (0) : {n_pred_real:>6}")
    print(f"    Predicted Fake (1) : {n_pred_fake:>6}")
    print(f"    Threshold used     : {args.threshold}")
    print(f"{'─'*50}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'─'*50}")
    print(f"  Validation Accuracy : {acc:.4f}")
    print(f"  Weighted F1-Score   : {f1:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"],
                                 zero_division=0))

    # ── 7. Save artefacts ────────────────────
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
