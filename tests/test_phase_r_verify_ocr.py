"""
Phase R.2 — live-OCR round-trip (slow; loads EasyOCR).

Skipped automatically when EasyOCR is unavailable. Asserts the honest contract:
verifying a freshly generated token WITHOUT supplying the serial never produces
a false TAMPERED — it is AUTHENTIC (serial read + pattern matched) or UNVERIFIED
(serial misread / unreadable). It must never accuse a genuine token of forgery
on the strength of an OCR guess.
"""

import cv2
import numpy as np
import pytest

from backend.security_pattern import secure_note_png
from backend.secure_note import verify_secure_note

pytestmark = pytest.mark.slow


def _reader_available():
    try:
        from backend.forensic import _get_easyocr_reader
        return _get_easyocr_reader() is not None
    except Exception:
        return False


@pytest.mark.skipif(not _reader_available(), reason="EasyOCR not available")
@pytest.mark.parametrize("serial", ["KKL7MP979885", "ABCDEFGH", "55667788"])
def test_ocr_roundtrip_never_false_tampered(serial):
    png = secure_note_png(serial, size=600)
    bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    r = verify_secure_note(bgr, serial=None)
    assert r["verdict"] in {"AUTHENTIC", "UNVERIFIED"}, r
    if r["verdict"] == "AUTHENTIC":
        # a match only happens if the serial was read correctly
        assert r["serial"] == serial
