import numpy as np
import os

import cv2
import pytest

from backend import polymer


def _polymer_like():
    img = np.full((180, 360, 3), (190, 145, 210), dtype=np.uint8)
    img[25:155, 265:320] = (245, 245, 245)
    img[40:140, 275:310] = (255, 255, 255)
    img[60:120, 60:140] = (250, 245, 220)
    return img


def _paper_like():
    return np.full((180, 360, 3), (185, 170, 150), dtype=np.uint8)


def test_transparent_window_passes_on_strong_candidate():
    out = polymer.detect_transparent_window(_polymer_like())
    assert out["status"] == "PASS"
    assert out["value"]["window_score"] > 0.95


def test_transparent_window_is_info_when_absent():
    out = polymer.detect_transparent_window(_paper_like())
    assert out["status"] == "INFO"


def test_sheen_never_false_fails():
    out = polymer.detect_substrate_sheen(_paper_like())
    assert out["status"] in {"PASS", "INFO"}


def test_analyze_polymer_features_combines_checks():
    out = polymer.analyze_polymer_features(_polymer_like())
    assert out["status"] in {"PASS", "INFO"}
    assert "transparent_window" in out["value"]
    assert "substrate_sheen" in out["value"]


@pytest.mark.parametrize(
    "rel_path",
    [
        "dataset/foreign/aud/full_note/real/aud_5_obverse.jpg",
        "dataset/foreign/cad/full_note/real/cad_20_obverse.png",
        "dataset/foreign/gbp/full_note/real/gbp_20_obverse.jpg",
        "dataset/foreign/php/full_note/real/php_1000_obverse.jpg",
    ],
)
def test_real_polymer_fixture_never_false_fails(rel_path):
    if not os.path.exists(rel_path):
        pytest.skip(f"polymer fixture missing: {rel_path}")
    img = cv2.imread(rel_path)
    assert img is not None
    out = polymer.analyze_polymer_features(img)
    assert out["status"] in {"PASS", "INFO"}
    assert out["value"]["transparent_window"]["status"] in {"PASS", "INFO"}
    assert out["value"]["substrate_sheen"]["status"] in {"PASS", "INFO"}
