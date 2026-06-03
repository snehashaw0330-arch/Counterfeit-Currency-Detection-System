"""
Phase G.3 — detected-note region overlay.

forensic.note_region(image) returns the located note's 4 corner points
normalised to [0,1] in ORIGINAL image coordinates (for the frontend overlay),
or None when no note is found. Deterministic, never raises — no model needed.
"""

import numpy as np
import pytest

from backend import forensic


def test_returns_four_normalised_points_for_a_note():
    import cv2
    img = cv2.imread("tests/sample_notes/real/real_100_obv.png")
    assert img is not None
    poly = forensic.note_region(img)
    assert poly is not None
    assert len(poly) == 4
    for x, y in poly:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_none_when_no_note():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (400, 800, 3), dtype=np.uint8)
    assert forensic.note_region(noise) is None


@pytest.mark.parametrize("bad", [None, np.zeros((1, 1, 3), np.uint8)])
def test_never_raises(bad):
    # Returns None (or a polygon) but never blows up.
    out = forensic.note_region(bad)
    assert out is None or len(out) == 4
