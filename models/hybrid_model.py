"""
Hybrid model that fuses CNN spatial features with FFT frequency features.

Architecture overview:
    ┌──────────────────────────────────────────────┐
    │  Spatial branch  (RGB image  — H×W×3)        │
    │    Conv2D stack → GlobalAveragePooling        │
    └─────────────────────┬────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────┐
    │  FFT branch  (magnitude spectrum — H×W×1)    │
    │    Conv2D stack → GlobalAveragePooling        │
    └─────────────────────┬────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────┐
    │  Fusion head                                  │
    │    Concatenate → Dense(256) → Dense(1,sigmoid)│
    └──────────────────────────────────────────────┘
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ─────────────────────────────────────────────
# Private branch builders
# ─────────────────────────────────────────────

def _spatial_branch(inputs, name_prefix: str = "spatial"):
    """Four-block CNN on a spatial (RGB) input."""
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


def _fft_branch(inputs, name_prefix: str = "fft"):
    """Three-block CNN on a single-channel FFT magnitude spectrum."""
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
    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    return x


# ─────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────

def build_hybrid_model(
    spatial_input_shape: tuple = (128, 128, 3),
    fft_input_shape: tuple = (128, 128, 1),
    dropout_rate: float = 0.5,
) -> Model:
    """
    Build and return the hybrid CNN + FFT model.

    The model has two named inputs:
    - ``"spatial_input"`` — float32 tensor with shape (H, W, 3)
    - ``"fft_input"``     — float32 tensor with shape (H, W, 1)

    Output is a scalar probability in [0, 1] (0 = Real, 1 = Fake).

    Args:
        spatial_input_shape: Shape of the spatial branch input.
        fft_input_shape:     Shape of the FFT branch input.
        dropout_rate:        Dropout probability in the fusion head.

    Returns:
        Compiled ``tf.keras.Model``.
    """
    # ── Inputs ───────────────────────────────
    spatial_input = layers.Input(
        shape=spatial_input_shape, name="spatial_input"
    )
    fft_input = layers.Input(shape=fft_input_shape, name="fft_input")

    # ── Branches ─────────────────────────────
    spatial_features = _spatial_branch(spatial_input, name_prefix="spatial")
    fft_features = _fft_branch(fft_input, name_prefix="fft")

    # ── Fusion ───────────────────────────────
    merged = layers.Concatenate(name="fusion")([spatial_features, fft_features])

    x = layers.Dense(512, activation="relu", name="fusion_fc1")(merged)
    x = layers.Dropout(dropout_rate, name="fusion_drop1")(x)
    x = layers.Dense(256, activation="relu", name="fusion_fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="fusion_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(
        inputs={"spatial_input": spatial_input, "fft_input": fft_input},
        outputs=outputs,
        name="hybrid_model",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model
