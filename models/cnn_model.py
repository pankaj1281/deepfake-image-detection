"""
CNN model for spatial feature extraction and binary classification.

Architecture:
    Conv2D(32) → ReLU → MaxPool
    Conv2D(64) → ReLU → MaxPool
    Conv2D(128) → ReLU → MaxPool
    GlobalAveragePooling → Dropout → Dense(256) → Dropout → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn_model(
    input_shape: tuple = (128, 128, 3),
    dropout_rate: float = 0.5,
) -> Model:
    """
    Build and return a CNN model for spatial fake-image detection.

    The model accepts a single spatial (RGB) input and outputs a scalar
    probability in [0, 1], where 0 ≈ Real and 1 ≈ Fake.

    Args:
        input_shape:  Shape of the spatial input (H, W, C).  Default: (128, 128, 3).
        dropout_rate: Dropout probability applied before the dense layers.

    Returns:
        Compiled ``tf.keras.Model``.
    """
    inputs = layers.Input(shape=input_shape, name="spatial_input")

    # ── Block 1 ──────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Block 2 ──────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Block 3 ──────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Block 4 ──────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Head ─────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_spatial_model")

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
