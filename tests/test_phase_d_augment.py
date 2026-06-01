"""Phase D — augmentation determinism / contract tests."""

import os

import cv2
import numpy as np

from backend import augment

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_notes")
REAL = os.path.join(SAMPLE_DIR, "real", "real_2000_obv.jpg")


def _load():
    img = cv2.imread(REAL)
    assert img is not None
    return img


def test_returns_requested_count():
    out = augment.augment_variants(_load(), 8, seed=1)
    assert len(out) == 8
    for v in out:
        assert v.ndim == 3 and v.shape[2] == 3
        assert v.dtype == np.uint8


def test_deterministic_same_seed():
    a = augment.augment_variants(_load(), 5, seed=7)
    b = augment.augment_variants(_load(), 5, seed=7)
    assert len(a) == len(b) == 5
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_seeds_differ():
    a = augment.augment_variants(_load(), 4, seed=1)
    b = augment.augment_variants(_load(), 4, seed=2)
    # At least one variant must differ between seeds.
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_variants_differ_from_original():
    img = _load()
    out = augment.augment_variants(img, 6, seed=3)
    # Augmentation must actually change the image.
    assert all(v.shape == img.shape for v in out)
    assert any(not np.array_equal(v, img) for v in out)


def test_never_raises_on_bad_input():
    assert augment.augment_variants(None, 3) == []
    assert augment.augment_variants(_load(), 0) == []
    tiny = np.zeros((8, 16, 3), dtype=np.uint8)
    assert len(augment.augment_variants(tiny, 2, seed=0)) == 2
