"""
predict.py — Run inference on a single image or a directory of images.

Usage (single image):
    python predict.py --image path/to/image.jpg --model models/best_hybrid_model.keras

Usage (batch / directory):
    python predict.py --input_dir path/to/images/ --model models/best_hybrid_model.keras
                      --output_dir predictions/

Options:
    --model       Path to a saved Keras model (.keras / .h5).
    --model_type  Architecture type: cnn | hybrid | efficientnet (default: hybrid).
    --image       Path to a single image file.
    --input_dir   Directory of images for batch prediction.
    --output_dir  Directory to save Grad-CAM heatmaps (default: predictions).
    --image_size  Resize images to IMAGE_SIZE × IMAGE_SIZE (default: 128).
    --threshold   Decision threshold (default: 0.5).
    --no_gradcam  Skip Grad-CAM heatmap generation.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.data_loader import load_single_image
from utils.grad_cam import GradCAM
from utils.metadata import MetadataAnalyzer


# ─────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether an image is Real or Fake."
    )
    parser.add_argument("--model", required=True,
                        help="Path to the saved Keras model.")
    parser.add_argument("--model_type", default="hybrid", choices=["cnn", "hybrid", "efficientnet"],
                        help="Model architecture type (default: hybrid).")
    parser.add_argument("--image", default=None,
                        help="Path to a single image for prediction.")
    parser.add_argument("--input_dir", default=None,
                        help="Directory of images for batch prediction.")
    parser.add_argument("--output_dir", default="predictions",
                        help="Directory to save heatmaps (default: predictions).")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Resize images to IMAGE_SIZE × IMAGE_SIZE.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5).")
    parser.add_argument("--no_gradcam", action="store_true",
                        help="Skip Grad-CAM heatmap generation.")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Core prediction helpers
# ─────────────────────────────────────────────

def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model not found: {model_path}")
    print(f"[INFO] Loading model from {model_path} …")
    return tf.keras.models.load_model(model_path)


def predict_single(
    model: tf.keras.Model,
    image_path: str,
    image_size: tuple,
    model_type: str,
    threshold: float,
    generate_gradcam: bool,
    output_dir: str,
) -> dict:
    """
    Run prediction for a single image.

    Args:
        model:            Loaded Keras model.
        image_path:       Path to the input image.
        image_size:       (height, width) tuple.
        model_type:       "cnn" or "hybrid".
        threshold:        Decision threshold.
        generate_gradcam: Whether to produce and save a Grad-CAM heatmap.
        output_dir:       Directory to save the heatmap image.

    Returns:
        Dictionary with keys: ``label``, ``confidence``, ``probability``,
        ``heatmap_path`` (None if not generated), ``metadata``.
    """
    spatial, fft = load_single_image(image_path, image_size)

    # Build model input
    if model_type in ("hybrid", "efficientnet"):
        model_input = {"spatial_input": spatial, "fft_input": fft}
    else:
        model_input = spatial

    # Inference
    probability = float(model.predict(model_input, verbose=0).flatten()[0])
    label = "Fake" if probability >= threshold else "Real"
    confidence = probability if label == "Fake" else 1.0 - probability

    # Metadata analysis
    meta_analyzer = MetadataAnalyzer(image_path)
    metadata = meta_analyzer.summary()

    # Grad-CAM
    heatmap_path = None
    if generate_gradcam:
        heatmap_path = _generate_and_save_heatmap(
            model=model,
            model_input=model_input,
            original_spatial=spatial[0],  # remove batch dim
            image_path=image_path,
            output_dir=output_dir,
            label=label,
            confidence=confidence,
        )

    return {
        "label": label,
        "confidence": confidence,
        "probability": probability,
        "heatmap_path": heatmap_path,
        "metadata": metadata,
    }


def _generate_and_save_heatmap(
    model,
    model_input,
    original_spatial: np.ndarray,
    image_path: str,
    output_dir: str,
    label: str,
    confidence: float,
) -> str:
    """Compute Grad-CAM and save an overlay figure.  Returns saved path."""
    os.makedirs(output_dir, exist_ok=True)

    grad_cam = GradCAM(model)
    heatmap = grad_cam.compute_heatmap(model_input)
    overlay = grad_cam.overlay_heatmap(
        heatmap=heatmap,
        original_image=(original_spatial * 255).astype(np.uint8),
    )

    # Build output figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Prediction: {label}  |  Confidence: {confidence:.2%}", fontsize=14)

    axes[0].imshow(original_spatial)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base}_gradcam.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Grad-CAM saved → {save_path}")
    return save_path


def print_result(image_path: str, result: dict) -> None:
    """Pretty-print prediction results to stdout."""
    print("\n" + "═" * 55)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  Raw Prob.  : {result['probability']:.4f}")
    print("─" * 55)
    m = result["metadata"]
    print(f"  Has EXIF   : {m['has_exif']}")
    print(f"  Camera     : {m['camera']}")
    print(f"  Software   : {m['software']}")
    print(f"  Meta Score : {m['manipulation_score']:.2f}")
    if result["heatmap_path"]:
        print(f"  Heatmap    : {result['heatmap_path']}")
    print("═" * 55)


# ─────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def predict_batch(
    model: tf.keras.Model,
    input_dir: str,
    image_size: tuple,
    model_type: str,
    threshold: float,
    generate_gradcam: bool,
    output_dir: str,
) -> list:
    """
    Run prediction on all supported images in a directory.

    Args:
        model:            Loaded Keras model.
        input_dir:        Directory containing image files.
        image_size:       (height, width) tuple.
        model_type:       "cnn" or "hybrid".
        threshold:        Decision threshold.
        generate_gradcam: Whether to produce Grad-CAM heatmaps.
        output_dir:       Directory to save heatmap images.

    Returns:
        List of result dicts (same format as ``predict_single``).
    """
    image_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        print(f"[WARNING] No supported images found in {input_dir}")
        return []

    results = []
    for img_path in image_files:
        result = predict_single(
            model=model,
            image_path=img_path,
            image_size=image_size,
            model_type=model_type,
            threshold=threshold,
            generate_gradcam=generate_gradcam,
            output_dir=output_dir,
        )
        print_result(img_path, result)
        results.append({"path": img_path, **result})

    # Summary
    n = len(results)
    n_fake = sum(1 for r in results if r["label"] == "Fake")
    print(f"\n[SUMMARY] Processed {n} images: {n_fake} Fake, {n - n_fake} Real")
    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if not args.image and not args.input_dir:
        sys.exit("[ERROR] Provide --image or --input_dir.")

    image_size = (args.image_size, args.image_size)
    model = load_model(args.model)

    if args.image:
        result = predict_single(
            model=model,
            image_path=args.image,
            image_size=image_size,
            model_type=args.model_type,
            threshold=args.threshold,
            generate_gradcam=not args.no_gradcam,
            output_dir=args.output_dir,
        )
        print_result(args.image, result)

    if args.input_dir:
        predict_batch(
            model=model,
            input_dir=args.input_dir,
            image_size=image_size,
            model_type=args.model_type,
            threshold=args.threshold,
            generate_gradcam=not args.no_gradcam,
            output_dir=args.output_dir,
        )
