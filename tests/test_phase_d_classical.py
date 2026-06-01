"""Phase D.5 — classical second-opinion inference tests.

The trained model binaries (models/classical/*.joblib) are
git-ignored and regenerable, so tests that need a trained model
SKIP gracefully when it is absent (fresh clone). The never-raises
contract is verified unconditionally.
"""

import os

import cv2
import numpy as np
import pytest

from backend import classical

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_notes")
REAL = os.path.join(SAMPLE_DIR, "real", "real_2000_obv.jpg")

_REQUIRED_KEYS = {"available", "name", "verdict", "confidence", "prob_genuine"}


def _load():
    img = cv2.imread(REAL)
    assert img is not None
    return img


def test_available_returns_bool():
    assert isinstance(classical.classical_available(), bool)


def test_predict_shape_when_trained():
    if not classical.classical_available():
        pytest.skip("classical models not trained (run scripts/train_classical.py)")
    out = classical.predict_classical(_load())
    assert out["available"] is True
    assert out["verdict"] in {"REAL", "FAKE"}
    assert 0.0 <= out["prob_genuine"] <= 1.0
    assert out["confidence"].endswith("%")
    assert out["name"]


def test_never_raises_on_garbage():
    cases = [
        None,
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.random.randint(0, 255, (30, 60, 3), dtype=np.uint8),
    ]
    for c in cases:
        out = classical.predict_classical(c)
        assert _REQUIRED_KEYS.issubset(out.keys())
