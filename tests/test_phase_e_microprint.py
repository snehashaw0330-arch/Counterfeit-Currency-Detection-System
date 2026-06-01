"""Phase E — micro-lettering / fine-print sharpness check tests.

Encodes the evidence-honest contract: FAIL on clearly-lost detail
(blur/blank) at adequate resolution, INFO when too low-res to
judge, and NEVER a PASS (calibration showed sharp fakes score as
high as genuine notes, so sharpness cannot certify authenticity).
"""

import os

import cv2
import numpy as np

from backend import forensic

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_notes")
HEAVY_BLUR = os.path.join(SAMPLE_DIR, "fake", "fake_04_heavy_blur.jpg")
LOWRES_PHONE = os.path.join(SAMPLE_DIR, "real_phone", "real_20_phone_specimen_obv.jpg")
SHARP_REAL = os.path.join(SAMPLE_DIR, "real", "real_2000_obv.jpg")


def _load(p):
    img = cv2.imread(p)
    assert img is not None, p
    return img


def test_fail_on_blurred_print():
    out = forensic.analyze_microprint(_load(HEAVY_BLUR))
    assert out["status"] == "FAIL"
    assert out["value"]["sharpness"] < forensic._MICROPRINT_FAIL


def test_info_on_low_resolution():
    out = forensic.analyze_microprint(_load(LOWRES_PHONE))
    assert out["status"] == "INFO"
    assert out["value"]["native_width"] < forensic._MICROPRINT_MIN_NATIVE_WIDTH


def test_never_certifies_sharp_note_as_pass():
    out = forensic.analyze_microprint(_load(SHARP_REAL))
    assert out["status"] != "PASS"
    assert out["status"] in {"INFO", "FAIL"}


def test_value_shape_and_never_raises():
    for c in [
        None,
        np.zeros((3, 3, 3), dtype=np.uint8),
        np.random.randint(0, 255, (40, 80, 3), dtype=np.uint8),
    ]:
        out = forensic.analyze_microprint(c)
        assert out["status"] in {"PASS", "FAIL", "INFO"}

    out = forensic.analyze_microprint(_load(SHARP_REAL))
    assert set(out["value"].keys()) == {"sharpness", "native_width"}
