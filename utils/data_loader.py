"""
Data loading, preprocessing, augmentation and FFT feature extraction utilities.
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_SIZE = (128, 128)  # (height, width) — can be overridden to (224, 224)
LABEL_REAL = 0
LABEL_FAKE = 1


# ─────────────────────────────────────────────
# FFT helpers
# ─────────────────────────────────────────────

def compute_fft_features(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to its 2-D FFT magnitude spectrum.

    The image is first converted to grayscale, the 2-D DFT is computed,
    shifted so that the zero-frequency component is centred, and the
    log-scaled magnitude spectrum is returned as a normalized float32
    array with shape (H, W, 1).

    Args:
        image: uint8 or float32 numpy array with shape (H, W, 3).

    Returns:
        Normalised FFT magnitude with shape (H, W, 1).
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))  # log-scale to compress dynamic range

    # Normalise to [0, 1]
    magnitude = magnitude / (magnitude.max() + 1e-8)
    return magnitude[..., np.newaxis].astype(np.float32)


# ─────────────────────────────────────────────
# Single-image loader (used by predict.py)
# ─────────────────────────────────────────────

def load_single_image(
    image_path: str,
    image_size: tuple = IMAGE_SIZE,
) -> tuple:
    """
    Load and preprocess a single image from disk.

    Args:
        image_path: Absolute or relative path to the image file.
        image_size: Target (height, width) tuple.

    Returns:
        Tuple of (spatial_input, fft_input) where each element is a
        float32 array with an added batch dimension:
        - spatial_input: shape (1, H, W, 3), pixel values in [0, 1]
        - fft_input:     shape (1, H, W, 1), normalized FFT magnitude
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size[1], image_size[0]))
    img_array = np.array(img, dtype=np.float32) / 255.0

    spatial = img_array[np.newaxis, ...]          # (1, H, W, 3)
    fft = compute_fft_features(img_array)[np.newaxis, ...]  # (1, H, W, 1)

    return spatial, fft


# ─────────────────────────────────────────────
# Dataset loader class
# ─────────────────────────────────────────────

class DataLoader:
    """
    Loads images from a directory that follows the structure:

        root/
          real/   ← images labelled as REAL (label 0)
          fake/   ← images labelled as FAKE (label 1)

    After calling ``load()``, the instance exposes:
        - x_train_spatial, x_val_spatial  : float32 (N, H, W, 3)
        - x_train_fft,     x_val_fft      : float32 (N, H, W, 1)
        - y_train,         y_val           : int32   (N,)

    Args:
        dataset_dir: Root directory containing *real/* and *fake/* sub-folders.
        image_size:  Target (height, width) for resizing. Default: (128, 128).
        test_size:   Fraction of data to reserve for validation. Default: 0.2.
        seed:        Random seed for reproducibility. Default: 42.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        dataset_dir: str,
        image_size: tuple = IMAGE_SIZE,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.test_size = test_size
        self.seed = seed

        # Populated after calling load()
        self.x_train_spatial = None
        self.x_val_spatial = None
        self.x_train_fft = None
        self.x_val_fft = None
        self.y_train = None
        self.y_val = None

    # ── private helpers ──────────────────────

    def _collect_paths(self) -> tuple:
        """Return (image_paths, labels) lists."""
        paths, labels = [], []
        for label, sub in [(LABEL_REAL, "real"), (LABEL_FAKE, "fake")]:
            folder = os.path.join(self.dataset_dir, sub)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Expected sub-folder not found: {folder}"
                )
            for fname in sorted(os.listdir(folder)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    paths.append(os.path.join(folder, fname))
                    labels.append(label)
        return paths, labels

    def _load_image(self, path: str) -> np.ndarray:
        """Load, resize and normalise a single image → float32 (H, W, 3)."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]))
        return np.array(img, dtype=np.float32) / 255.0

    # ── public API ───────────────────────────

    def load(self, verbose: bool = True) -> "DataLoader":
        """
        Load all images, compute FFT features, split into train/val.

        Args:
            verbose: Print progress if True.

        Returns:
            self (for method chaining).
        """
        paths, labels = self._collect_paths()
        n = len(paths)
        if n == 0:
            raise ValueError(
                f"No supported images found in {self.dataset_dir}. "
                "Make sure the dataset/real and dataset/fake folders are populated."
            )

        if verbose:
            print(f"Found {n} images — loading …")

        spatial_images = []
        fft_images = []

        for i, path in enumerate(paths):
            img = self._load_image(path)
            spatial_images.append(img)
            fft_images.append(compute_fft_features(img))

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{n}")

        spatial = np.stack(spatial_images, axis=0)  # (N, H, W, 3)
        fft = np.stack(fft_images, axis=0)           # (N, H, W, 1)
        labels_arr = np.array(labels, dtype=np.int32)

        # Train / validation split (stratified)
        indices = np.arange(n)
        tr_idx, va_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=labels_arr,
        )

        self.x_train_spatial = spatial[tr_idx]
        self.x_val_spatial = spatial[va_idx]
        self.x_train_fft = fft[tr_idx]
        self.x_val_fft = fft[va_idx]
        self.y_train = labels_arr[tr_idx]
        self.y_val = labels_arr[va_idx]

        if verbose:
            print(
                f"Dataset loaded: {len(tr_idx)} train / {len(va_idx)} val samples"
            )

        return self

    def get_tf_datasets(
        self,
        batch_size: int = 32,
        augment_train: bool = True,
    ) -> tuple:
        """
        Build ``tf.data.Dataset`` objects for the training and validation sets.

        Each dataset yields ``((spatial, fft), label)`` batches.

        Args:
            batch_size:    Number of samples per batch.
            augment_train: Apply random augmentation to the training set.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if self.x_train_spatial is None:
            raise RuntimeError("Call .load() before .get_tf_datasets()")

        def _make_ds(spatial, fft, labels, shuffle: bool, augment: bool):
            ds = tf.data.Dataset.from_tensor_slices(
                ({"spatial_input": spatial, "fft_input": fft}, labels)
            )
            if shuffle:
                ds = ds.shuffle(len(labels), seed=self.seed)
            if augment:
                ds = ds.map(
                    _augment_sample, num_parallel_calls=tf.data.AUTOTUNE
                )
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = _make_ds(
            self.x_train_spatial,
            self.x_train_fft,
            self.y_train,
            shuffle=True,
            augment=augment_train,
        )
        val_ds = _make_ds(
            self.x_val_spatial,
            self.x_val_fft,
            self.y_val,
            shuffle=False,
            augment=False,
        )
        return train_ds, val_ds


# ─────────────────────────────────────────────
# Augmentation helper (applied inside tf.data)
# ─────────────────────────────────────────────

def _augment_sample(inputs: dict, label):
    """
    Apply random spatial augmentations to a single training sample.

    Augmentations applied:
    - Random horizontal flip
    - Random vertical flip
    - Random rotation (±15°)
    - Random brightness / contrast jitter

    Note: FFT features are re-computed from the augmented spatial image
    so both branches stay consistent.
    """
    spatial = inputs["spatial_input"]  # (H, W, 3)

    # Random flips
    spatial = tf.image.random_flip_left_right(spatial)
    spatial = tf.image.random_flip_up_down(spatial)

    # Random rotation (±15°)
    spatial = tf.keras.layers.RandomRotation(factor=15 / 360)(
        tf.expand_dims(spatial, 0)
    )[0]

    # Random brightness / contrast
    spatial = tf.image.random_brightness(spatial, max_delta=0.1)
    spatial = tf.image.random_contrast(spatial, lower=0.9, upper=1.1)
    spatial = tf.clip_by_value(spatial, 0.0, 1.0)

    # Re-compute FFT from augmented image
    fft = tf.numpy_function(
        func=lambda x: compute_fft_features(x.numpy()),
        inp=[spatial],
        Tout=tf.float32,
    )
    fft.set_shape(inputs["fft_input"].shape)

    return {"spatial_input": spatial, "fft_input": fft}, label
