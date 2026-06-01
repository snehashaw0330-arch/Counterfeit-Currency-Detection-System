"""Phase D.2 — hand-crafted feature-vector extractor tests.

Validates the contract in backend/features.py: fixed length,
deterministic, finite, never raises, and actually carries signal
(a real note must not featurise identically to a blank frame).
"""

import os

import cv2
import numpy as np

from backend import features

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_notes")
REAL = os.path.join(SAMPLE_DIR, "real", "real_2000_obv.jpg")
BLANK = os.path.join(SAMPLE_DIR, "fake", "fake_01_blank_black.jpg")


def _load(path):
    img = cv2.imread(path)
    assert img is not None, f"fixture missing: {path}"
    return img


def test_feature_dim_matches_names():
    assert features.FEATURE_DIM == len(features.FEATURE_NAMES)
    assert features.FEATURE_DIM == 50


def test_vector_shape_and_dtype():
    vec = features.extract_feature_vector(_load(REAL))
    assert vec.shape == (features.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_all_finite_on_real_and_blank():
    for p in (REAL, BLANK):
        vec = features.extract_feature_vector(_load(p))
        assert np.all(np.isfinite(vec)), f"non-finite feature on {p}"


def test_deterministic():
    img = _load(REAL)
    a = features.extract_feature_vector(img)
    b = features.extract_feature_vector(img)
    assert np.array_equal(a, b)


def test_never_raises_on_garbage():
    cases = [
        None,
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((5, 5), dtype=np.uint8),            # 2-D grayscale
        np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8),
    ]
    for c in cases:
        vec = features.extract_feature_vector(c)
        assert vec.shape == (features.FEATURE_DIM,)
        assert np.all(np.isfinite(vec))


def test_carries_signal_real_vs_blank():
    real = features.extract_feature_vector(_load(REAL))
    blank = features.extract_feature_vector(_load(BLANK))

    # A real note and an all-black frame must not featurise the same.
    assert not np.array_equal(real, blank)

    # The real note carries colour the blank cannot.
    sat_idx = features.FEATURE_NAMES.index("mean_sat")
    assert real[sat_idx] > blank[sat_idx]
