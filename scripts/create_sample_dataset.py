"""
scripts/create_sample_dataset.py — Generate a small synthetic dataset for testing.

This script creates simple synthetic images (solid colors, gradients, shapes)
that let you verify the full training and inference pipeline without needing a
real deepfake dataset.

Usage (run from the project root):
    python scripts/create_sample_dataset.py
    python scripts/create_sample_dataset.py --n_per_class 50 --image_size 128

The output is placed in:
    dataset/real/   ← synthetic "real" images (smooth gradients / uniform patches)
    dataset/fake/   ← synthetic "fake" images (noisy / high-frequency patterns)

Once generated you can train immediately:
    python train.py --dataset_dir dataset --model_type hybrid --epochs 5
"""

import argparse
import os
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit(
        "[ERROR] Pillow is not installed. Run:  pip install Pillow"
    )


# ─────────────────────────────────────────────
# Image generators
# ─────────────────────────────────────────────

def _random_gradient(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a smooth color-gradient image (simulates 'real' appearance)."""
    # Pick two random colors and interpolate along one axis
    c1 = rng.integers(30, 230, size=3).astype(np.float32)
    c2 = rng.integers(30, 230, size=3).astype(np.float32)
    # t: (size, 1) so it broadcasts across 3 color channels
    t = np.linspace(0, 1, size, dtype=np.float32).reshape(size, 1)
    row = ((1 - t) * c1 + t * c2).astype(np.uint8)  # (size, 3)
    if rng.random() > 0.5:
        # Horizontal gradient: each row is the same blend
        img = np.broadcast_to(row[:, np.newaxis, :], (size, size, 3)).copy()
    else:
        # Vertical gradient: each column is the same blend
        img = np.broadcast_to(row[np.newaxis, :, :], (size, size, 3)).copy()
    return img


def _random_noisy(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a high-frequency noisy image (simulates 'fake' GAN artifacts)."""
    base = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    # Add structured grid-like noise to mimic GAN checkerboard artifacts
    grid = np.zeros((size, size, 3), dtype=np.int16)
    block = max(4, size // 16)
    for i in range(0, size, block * 2):
        for j in range(0, size, block * 2):
            val = int(rng.integers(50, 150))
            grid[i : i + block, j : j + block] = val
    noisy = np.clip(base.astype(np.int16) + grid, 0, 255).astype(np.uint8)
    return noisy


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def generate(n_per_class: int, image_size: int, dataset_dir: str, seed: int) -> None:
    rng = np.random.default_rng(seed)

    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"[INFO] Generating {n_per_class} synthetic real images → {real_dir}")
    for i in range(n_per_class):
        arr = _random_gradient(image_size, rng)
        Image.fromarray(arr).save(os.path.join(real_dir, f"real_{i:04d}.png"))

    print(f"[INFO] Generating {n_per_class} synthetic fake images → {fake_dir}")
    for i in range(n_per_class):
        arr = _random_noisy(image_size, rng)
        Image.fromarray(arr).save(os.path.join(fake_dir, f"fake_{i:04d}.png"))

    total = n_per_class * 2
    print(
        f"\n✅  Done — {total} images created "
        f"({n_per_class} real + {n_per_class} fake).\n"
        "You can now train the model:\n"
        "  python train.py --dataset_dir dataset --model_type hybrid --epochs 5"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic dataset for pipeline testing."
    )
    parser.add_argument(
        "--n_per_class", type=int, default=100,
        help="Number of images to generate per class (default: 100).",
    )
    parser.add_argument(
        "--image_size", type=int, default=128,
        help="Width and height of each generated image in pixels (default: 128).",
    )
    parser.add_argument(
        "--dataset_dir", default="dataset",
        help="Root dataset directory (default: dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        n_per_class=args.n_per_class,
        image_size=args.image_size,
        dataset_dir=args.dataset_dir,
        seed=args.seed,
    )
