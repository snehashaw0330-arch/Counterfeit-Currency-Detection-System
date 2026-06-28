"""
Phase R.2 — closed-loop secure-note verification.

Fast + deterministic: the three-band verdict logic and the OCR-trust rule are
exercised by monkeypatching the similarity/serial helpers (no EasyOCR load), and
the real authentic/tampered cases run on generated tokens (no model load). A
live-OCR round-trip is covered in test_phase_r_verify_ocr.py (skipped when
EasyOCR is unavailable).
"""

import cv2
import numpy as np

from backend.security_pattern import secure_note_png, serial_token
from backend.secure_note import (
    verify_secure_note,
    _MATCH_THRESHOLD,
    _TAMPER_THRESHOLD,
)

import backend.secure_note as sn


def _decode(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def _token(serial, size=600):
    return _decode(secure_note_png(serial, size))


# --------------------------------------------------------------------------
# Three-band verdict logic, trusted (caller-supplied) serial
# --------------------------------------------------------------------------

def test_band_authentic(monkeypatch):
    monkeypatch.setattr(sn, "_pattern_similarity", lambda bgr, norm: 0.90)
    r = verify_secure_note(_token("ABC123"), serial="ABC123")
    assert r["verdict"] == "AUTHENTIC" and r["match"] is True


def test_band_tampered(monkeypatch):
    monkeypatch.setattr(sn, "_pattern_similarity", lambda bgr, norm: 0.30)
    r = verify_secure_note(_token("ABC123"), serial="ABC123")
    assert r["verdict"] == "TAMPERED" and r["match"] is False


def test_band_unverified_midrange(monkeypatch):
    mid = (_MATCH_THRESHOLD + _TAMPER_THRESHOLD) / 2
    monkeypatch.setattr(sn, "_pattern_similarity", lambda bgr, norm: mid)
    r = verify_secure_note(_token("ABC123"), serial="ABC123")
    assert r["verdict"] == "UNVERIFIED" and r["match"] is False


# --------------------------------------------------------------------------
# OCR-trust rule: an unconfirmed OCR'd serial never yields TAMPERED
# --------------------------------------------------------------------------

def test_ocr_match_is_authentic(monkeypatch):
    monkeypatch.setattr(sn, "_read_serial", lambda bgr: "ABC123")
    monkeypatch.setattr(sn, "_pattern_similarity", lambda bgr, norm: 0.90)
    r = verify_secure_note(_token("ABC123"), serial=None)
    assert r["verdict"] == "AUTHENTIC"
    assert r["serial_source"] == "ocr"


def test_ocr_nonmatch_is_unverified_not_tampered(monkeypatch):
    # OCR misread the serial -> pattern won't match -> must NOT accuse (TAMPERED)
    monkeypatch.setattr(sn, "_read_serial", lambda bgr: "WRONGXYZ")
    monkeypatch.setattr(sn, "_pattern_similarity", lambda bgr, norm: 0.30)
    r = verify_secure_note(_token("ABC123"), serial=None)
    assert r["verdict"] == "UNVERIFIED"
    assert r["serial_source"] == "ocr"


def test_unverified_when_ocr_blank(monkeypatch):
    monkeypatch.setattr(sn, "_read_serial", lambda bgr: None)
    r = verify_secure_note(_token("ABC123"), serial=None)
    assert r["verdict"] == "UNVERIFIED"
    assert r["serial"] is None


# --------------------------------------------------------------------------
# Real authentic / tampered cases (generated tokens, wide SSIM margins)
# --------------------------------------------------------------------------

def test_authentic_when_serial_matches():
    r = verify_secure_note(_token("KKL7MP979885"), serial="KKL7MP979885")
    assert r["verdict"] == "AUTHENTIC"
    assert r["similarity"] >= _MATCH_THRESHOLD
    assert r["serial"] == "KKL7MP979885"
    assert r["serial_source"] == "input"
    assert r["code"] == serial_token("KKL7MP979885")


def test_authentic_survives_jpeg():
    tok = _token("8AC123456")
    ok, enc = cv2.imencode(".jpg", tok, [cv2.IMWRITE_JPEG_QUALITY, 55])
    r = verify_secure_note(cv2.imdecode(enc, cv2.IMREAD_COLOR), serial="8AC123456")
    assert r["verdict"] == "AUTHENTIC"


def test_tampered_on_wrong_serial():
    r = verify_secure_note(_token("ABCD5678"), serial="ZZ9 000001")
    assert r["verdict"] == "TAMPERED"
    assert r["similarity"] <= _TAMPER_THRESHOLD


def test_tampered_on_swapped_pattern():
    tok_a = _token("AAAA1111")
    tok_b = _token("BBBB2222")
    h, w = tok_a.shape[:2]
    swapped = tok_a.copy()
    swapped[h - w:h, 0:w] = tok_b[h - w:h, 0:w]  # header says A, disc is B
    r = verify_secure_note(swapped, serial="AAAA1111")
    assert r["verdict"] == "TAMPERED"


def test_low_resolution_is_never_falsely_authenticated():
    tok = _token("7QP441122", size=600)
    half = cv2.resize(tok, (tok.shape[1] // 2, tok.shape[0] // 2))
    r = verify_secure_note(half, serial="7QP441122")
    assert r["verdict"] != "AUTHENTIC"
    assert r["similarity"] < _MATCH_THRESHOLD


# --------------------------------------------------------------------------
# Robustness / contract
# --------------------------------------------------------------------------

def test_never_raises_on_garbage():
    junk = np.random.randint(0, 255, (40, 90, 3), dtype=np.uint8)
    r = verify_secure_note(junk, serial="ABC123")
    assert r["verdict"] in {"AUTHENTIC", "TAMPERED", "UNVERIFIED"}


def test_response_shape():
    r = verify_secure_note(_token("ABC123"), serial="ABC123")
    for k in ("verdict", "serial", "serial_source", "similarity", "threshold",
              "match", "code", "note"):
        assert k in r
