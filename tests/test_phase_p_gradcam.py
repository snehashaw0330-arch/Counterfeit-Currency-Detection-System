"""
Phase P.2 — Grad-CAM heatmap (graceful-contract tests).

Grad-CAM is best-effort: it must return None (never raise) when it can't build a
gradient model, so /predict is never broken by it. We don't load the real model
here (that's exercised end-to-end); we assert the never-raises contract.
"""

import numpy as np

from backend import gradcam


def test_find_last_conv_none_on_plain_object():
    class Dummy:
        pass
    assert gradcam._find_last_conv(Dummy()) is None


def test_find_last_conv_none_on_empty_layers():
    class Dummy:
        layers = []
    assert gradcam._find_last_conv(Dummy()) is None


def test_compute_heatmap_none_on_bad_model():
    x = np.zeros((1, 224, 224, 3), dtype="float32")
    # A non-Keras object has no conv layers -> graceful None, no exception.
    assert gradcam.compute_heatmap(object(), x) is None
    assert gradcam.compute_heatmap(None, x) is None
