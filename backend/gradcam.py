"""
Phase P.2 — Grad-CAM heatmap.

Produces a class-activation heatmap showing WHERE the MobileNetV2 classifier
looked when scoring the note — an explainability layer for the deep model.

Strictly best-effort: Grad-CAM on a saved/nested Keras model can fail (graph
disconnected, no conv layer found, etc.). Every path is wrapped so a failure
returns None and the /predict response simply omits the heatmap — it never
breaks a prediction. Pure numpy out; TensorFlow is imported lazily.
"""

import numpy as np


def _find_last_conv(model):
    """Depth-first search for the last 4-D conv layer (handles a nested base
    model such as MobileNetV2 wrapped inside the classifier)."""
    found = []

    def walk(m):
        for layer in getattr(m, "layers", []):
            if getattr(layer, "layers", None):
                walk(layer)
            name = layer.__class__.__name__.lower()
            if "conv" in name:
                try:
                    if len(layer.output.shape) == 4:
                        found.append(layer)
                except Exception:
                    pass

    walk(model)
    return found[-1] if found else None


def compute_heatmap(model, x):
    """x: (1, 224, 224, 3) float array in [0, 1].
    Returns a small HxW float heatmap in [0, 1], or None on any failure."""
    try:
        import tensorflow as tf

        conv = _find_last_conv(model)
        if conv is None:
            return None

        grad_model = tf.keras.models.Model(
            model.inputs, [conv.output, model.outputs[0]]
        )

        x_t = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x_t, training=False)
            # Binary sigmoid head → a single score to attribute.
            score = preds[:, 0]

        grads = tape.gradient(score, conv_out)
        if grads is None:
            return None

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))      # per-channel
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # weighted sum
        cam = tf.nn.relu(cam)
        peak = tf.reduce_max(cam)
        if float(peak) <= 0:
            return None
        cam = cam / (peak + 1e-8)
        return cam.numpy().astype("float32")
    except Exception:
        return None
