"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Grad-CAM highlights the image regions that most influenced the model's
prediction, making CNN decisions interpretable.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import cv2
import tensorflow as tf


class GradCAM:
    """
    Generates Grad-CAM heatmaps for a given Keras model and input image.

    Args:
        model:      A compiled Keras model.
        layer_name: Name of the target convolutional layer whose activations
                    are used to produce the heatmap.  If None, the last Conv2D
                    layer in the model is located automatically.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()

    # ── private helpers ──────────────────────

    def _find_last_conv_layer(self) -> str:
        """Return the name of the last Conv2D layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError(
            "No Conv2D layer found in the model. "
            "Please specify a layer_name explicitly."
        )

    # ── public API ───────────────────────────

    def compute_heatmap(
        self,
        inputs,
        class_index: int = None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for the given input.

        Args:
            inputs:      Model input(s) — a single numpy array or a list/dict
                         of arrays (for multi-input models).
            class_index: Target class index.  If None, the class with the
                         highest predicted probability is used.
            eps:         Small constant added before normalisation.

        Returns:
            Float32 numpy array with shape (H, W) containing the normalised
            heatmap, where values are in [0, 1].
        """
        # Build a sub-model that outputs (feature_maps, predictions)
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output,
            ],
        )

        # Record operations for auto-differentiation
        with tf.GradientTape() as tape:
            if isinstance(inputs, dict):
                tensor_inputs = {
                    k: tf.cast(v, tf.float32) for k, v in inputs.items()
                }
            elif isinstance(inputs, (list, tuple)):
                tensor_inputs = [tf.cast(x, tf.float32) for x in inputs]
            else:
                tensor_inputs = tf.cast(inputs, tf.float32)

            conv_outputs, predictions = grad_model(tensor_inputs)

            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))

            # Scalar output for the chosen class
            loss = predictions[:, class_index]

        # Gradients of the loss w.r.t. the conv feature maps
        grads = tape.gradient(loss, conv_outputs)  # (1, h, w, C)

        # Global-average-pool the gradients over the spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weight the feature maps by the pooled gradients
        conv_outputs = conv_outputs[0]                          # (h, w, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap = tf.squeeze(heatmap)                           # (h, w)

        # ReLU + normalise
        heatmap = tf.nn.relu(heatmap).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        return heatmap.astype(np.float32)

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on top of the original image.

        Args:
            heatmap:        2-D float32 array from ``compute_heatmap``.
            original_image: uint8 numpy array with shape (H, W, 3) in RGB.
            alpha:          Blending factor for the heatmap overlay.
            colormap:       OpenCV colormap constant.  Default: COLORMAP_JET.

        Returns:
            uint8 numpy array with shape (H, W, 3) in RGB.
        """
        h, w = original_image.shape[:2]

        # Resize heatmap to match the original image
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to colour (BGR) and then to RGB
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with the original image
        if original_image.dtype != np.uint8:
            original_image = np.uint8(original_image * 255)

        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
        return overlay
