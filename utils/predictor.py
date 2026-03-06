"""
predictor.py
Inference utilities: preprocessing, prediction, confidence, Grad-CAM heatmap.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import io


CLASS_NAMES_CLEAN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and batch-expand an image for inference."""
    img = image.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)


def predict_multiclass(model, image: Image.Image) -> dict:
    """
    Returns:
        {
          "class": str,          # predicted class name
          "emoji": str,          # emoji
          "confidence": float,   # 0-1
          "probabilities": list  # per-class probabilities
        }
    """
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]  # shape (10,)
    idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES_CLEAN[idx],
        "emoji": CLASS_EMOJIS[idx],
        "confidence": float(probs[idx]),
        "probabilities": probs.tolist(),
        "class_index": idx,
    }


def predict_binary(model, image: Image.Image) -> dict:
    """
    Binary: airplane (class 0) vs. rest.
    Returns:
        {
          "label": str,
          "is_airplane": bool,
          "confidence": float,   # confidence in the predicted label
          "airplane_prob": float # raw P(airplane)
        }
    """
    arr = preprocess_image(image)
    airplane_prob = float(model.predict(arr, verbose=0)[0][0])
    is_airplane = airplane_prob > 0.5
    confidence = airplane_prob if is_airplane else (1.0 - airplane_prob)
    return {
        "label": "Airplane" if is_airplane else "Non-Airplane",
        "is_airplane": is_airplane,
        "confidence": confidence,
        "airplane_prob": airplane_prob,
    }


def get_gradcam_heatmap(model, image: Image.Image, layer_name: str = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the predicted class.
    Returns a (32, 32) numpy array normalised to [0, 1].
    """
    arr = preprocess_image(image)  # (1, 32, 32, 3)

    # Auto-detect the last Conv layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    if layer_name is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(arr, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Normalise
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize to 32x32
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((32, 32), Image.LANCZOS)
        return np.array(heatmap_img) / 255.0

    except Exception:
        return None


def overlay_gradcam(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay Grad-CAM heatmap on the original image."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Colorize heatmap
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap)[:, :, :3]  # drop alpha
    heatmap_img = Image.fromarray((colored * 255).astype(np.uint8)).resize(
        original_image.size, Image.LANCZOS
    )

    # Blend
    original_rgb = original_image.convert("RGB")
    blended = Image.blend(original_rgb, heatmap_img, alpha=alpha)
    return blended
