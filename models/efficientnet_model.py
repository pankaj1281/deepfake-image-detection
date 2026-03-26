"""
Advanced hybrid model: EfficientNetB0 (pretrained spatial backbone) + FFT CNN branch.

Architecture overview:
    ┌─────────────────────────────────────────────────────┐
    │  Spatial branch  (RGB image — H×W×3)                │
    │    EfficientNetB0 (ImageNet weights, optional       │
    │    fine-tuning) → GlobalAveragePooling → Dense(256) │
    └──────────────────────────┬──────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │  FFT branch  (magnitude spectrum — H×W×1)           │
    │    4-block Conv2D + GAP → Dense(256)                │
    └──────────────────────────┬──────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │  Fusion head                                        │
    │    Concatenate → Dense(512) → BN → Dropout          │
    │    → SE attention → Dense(256) → Dense(1, sigmoid)  │
    └─────────────────────────────────────────────────────┘

Using a pretrained EfficientNetB0 backbone dramatically improves feature
extraction quality, especially when the training dataset is small.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _build_fft_branch(inputs, name_prefix: str = "fft"):
    """Enhanced 4-block CNN for the single-channel FFT magnitude input."""
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name=f"{name_prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                      name=f"{name_prefix}_conv4")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn4")(x)
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    return x


def _se_block(inputs, reduction: int = 16, name_prefix: str = "se"):
    """
    Squeeze-and-Excitation (SE) channel attention block for 1-D feature vectors.

    Recalibrates channel-wise feature responses by learning per-channel
    scaling factors conditioned on the global channel statistics.
    """
    channels = inputs.shape[-1]
    se = layers.Dense(max(channels // reduction, 1), activation="relu",
                      name=f"{name_prefix}_fc1")(inputs)
    se = layers.Dense(channels, activation="sigmoid",
                      name=f"{name_prefix}_fc2")(se)
    return layers.Multiply(name=f"{name_prefix}_scale")([inputs, se])


# ─────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────

def build_efficientnet_hybrid_model(
    spatial_input_shape: tuple = (224, 224, 3),
    fft_input_shape: tuple = (224, 224, 1),
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    label_smoothing: float = 0.1,
) -> Model:
    """
    Build and return the EfficientNetB0-based hybrid model.

    The model has two named inputs — identical to the standard hybrid model
    so it is fully compatible with the existing training / inference pipeline:
    - ``"spatial_input"`` — float32 tensor with shape (H, W, 3)
    - ``"fft_input"``     — float32 tensor with shape (H, W, 1)

    Output is a scalar probability in [0, 1] (0 = Real, 1 = Fake).

    Args:
        spatial_input_shape: Shape of the spatial branch input (H, W, 3).
        fft_input_shape:     Shape of the FFT branch input (H, W, 1).
        dropout_rate:        Dropout probability in the fusion head.
        freeze_backbone:     When True, EfficientNetB0 weights are frozen
                             (useful for phase-1 of two-phase training).
        label_smoothing:     Label smoothing coefficient for the loss.

    Returns:
        Compiled ``tf.keras.Model``.
    """
    # ── Inputs ───────────────────────────────
    spatial_input = layers.Input(shape=spatial_input_shape, name="spatial_input")
    fft_input = layers.Input(shape=fft_input_shape, name="fft_input")

    # ── Spatial branch: EfficientNetB0 ───────
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=spatial_input_shape,
        name="efficientnetb0_backbone",
    )
    backbone.trainable = not freeze_backbone

    x_spatial = backbone(spatial_input, training=not freeze_backbone)
    x_spatial = layers.GlobalAveragePooling2D(name="spatial_gap")(x_spatial)
    x_spatial = layers.Dense(256, activation="relu", name="spatial_proj")(x_spatial)
    x_spatial = layers.Dropout(dropout_rate * 0.5, name="spatial_drop")(x_spatial)

    # ── FFT branch ───────────────────────────
    x_fft = _build_fft_branch(fft_input, name_prefix="fft")
    x_fft = layers.Dense(256, activation="relu", name="fft_proj")(x_fft)
    x_fft = layers.Dropout(dropout_rate * 0.5, name="fft_drop")(x_fft)

    # ── Fusion head ──────────────────────────
    merged = layers.Concatenate(name="fusion")([x_spatial, x_fft])  # (512,)

    x = layers.Dense(512, activation="relu", name="fusion_fc1")(merged)
    x = layers.BatchNormalization(name="fusion_bn1")(x)
    x = layers.Dropout(dropout_rate, name="fusion_drop1")(x)

    # Channel-wise SE attention on the fused features
    x = _se_block(x, reduction=16, name_prefix="fusion_se")

    x = layers.Dense(256, activation="relu", name="fusion_fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="fusion_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(
        inputs={"spatial_input": spatial_input, "fft_input": fft_input},
        outputs=outputs,
        name="efficientnet_hybrid_model",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model
