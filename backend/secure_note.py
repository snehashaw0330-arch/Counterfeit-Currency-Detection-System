"""
Phase R.2 — closed-loop secure-note verification.

Given an image of a SECURE-NOTE TOKEN produced by `backend.security_pattern`
(Phase R.1), verify it by REGENERATING the guilloché the serial should produce
and comparing it to the guilloché EXTRACTED from the image (structural
similarity, at the disc's native scale). This is an honest **closed loop**: it
verifies tokens *we* generate (which deterministically bind a serial to its
guilloché), NOT real banknotes — real notes carry no serial-derived guilloché.

The serial may be supplied by the caller (primary, trusted path) or, if omitted,
OCR'd from the token header. OCR is best-effort and cannot be fully trusted: a
single misread character regenerates a different pattern, so an OCR'd serial is
NEVER used to accuse a token of being TAMPERED — it can only yield AUTHENTIC
(a pattern match proves the serial was read correctly) or UNVERIFIED ("enter the
serial"). A TAMPERED verdict is only returned when the serial is trusted.

Verdicts:
  AUTHENTIC   — the extracted guilloché matches the pattern the serial produces.
  TAMPERED    — trusted serial, but the guilloché clearly does not match it.
  UNVERIFIED  — no serial could be read, the OCR'd serial couldn't be confirmed,
                or the image is too degraded to decide.

The printed CODE (serial_token) is a short human-readable fingerprint of the
serial; it is returned as `code` for reference. Never raises.
"""

import re

import numpy as np
import cv2
from skimage.metrics import structural_similarity

from backend.security_pattern import (
    generate_pattern,
    normalize_serial,
    serial_token,
)

# Three-band verdict on the SSIM between the extracted disc and the disc the
# serial should produce, compared at the disc's NATIVE scale (R.2 calibration):
#   >= _MATCH_THRESHOLD   -> AUTHENTIC  (pristine/JPEG genuine ~0.73-0.81)
#   <= _TAMPER_THRESHOLD  -> TAMPERED   (different serial's pattern <=0.45)
#   in between            -> UNVERIFIED (detail too degraded, e.g. low-res)
# AUTHENTIC needs >=0.55 and the worst forgery measured was 0.447 — a ~0.10
# margin, so a wrong pattern is never falsely authenticated; the inconclusive
# band is the safe failure direction.
_MATCH_THRESHOLD = 0.55
_TAMPER_THRESHOLD = 0.45
# Comparison scale is the disc's own width, clamped to this range (cap keeps a
# huge upload cheap; floor keeps tiny crops meaningful).
_MIN_CMP = 160
_MAX_CMP = 1024


def _gray(arr_bgr):
    """BGR (or single-channel) array -> grayscale (no resize)."""
    arr = np.asarray(arr_bgr)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return arr.astype(np.uint8)


def _disc_bgr(bgr):
    """The guilloché disc of an uploaded token: the bottom WxW square (token
    layout = header band over a square disc). If the image is already ~square
    (disc only), the whole image is used."""
    h, w = bgr.shape[:2]
    return bgr[h - w:h, 0:w] if h > w else bgr


def _pattern_similarity(bgr, norm):
    """SSIM between the uploaded guilloché disc and the disc `norm` SHOULD
    produce, compared at the disc's native scale (so 1px line-work aligns)."""
    disc = _disc_bgr(bgr)
    width = disc.shape[1]
    size = int(min(max(width, _MIN_CMP), _MAX_CMP))
    extracted = cv2.resize(_gray(disc), (size, size), interpolation=cv2.INTER_AREA)
    expected = np.asarray(generate_pattern(norm, size).convert("L"), dtype=np.uint8)
    return float(structural_similarity(expected, extracted, data_range=255))


def _read_serial(bgr):
    """Best-effort OCR of the token header -> the serial string, or None.

    Reuses forensic's EasyOCR singleton (no second reader). Never raises."""
    try:
        from backend.forensic import _get_easyocr_reader
        reader = _get_easyocr_reader()
        if reader is None:
            return None
        tokens = reader.readtext(bgr, detail=0)
    except Exception:
        return None
    text = re.sub(r"[^A-Z0-9 ]", " ", " ".join(str(t).upper() for t in tokens))
    m = re.search(r"SERIAL\s+([A-Z0-9]+)", text)
    return m.group(1) if m else None


def verify_secure_note(bgr, serial=None):
    """Verify a Phase R.1 secure-note token. Returns a dict; never raises."""
    result = {
        "verdict": "UNVERIFIED",
        "serial": None,
        "serial_source": None,
        "similarity": None,
        "threshold": _MATCH_THRESHOLD,
        "match": False,
        "code": None,
        "note": "",
    }
    try:
        source = "input"
        if serial is None or not str(serial).strip():
            serial = _read_serial(bgr)
            source = "ocr"

        if not serial or not normalize_serial(serial):
            result["note"] = ("Could not determine a serial number (none "
                              "supplied and OCR could not read one).")
            return result

        norm = normalize_serial(serial)
        result["serial"] = norm
        result["serial_source"] = source
        result["code"] = serial_token(norm)

        score = max(0.0, min(1.0, _pattern_similarity(bgr, norm)))
        result["similarity"] = round(score, 4)

        if score >= _MATCH_THRESHOLD:
            result["match"] = True
            result["verdict"] = "AUTHENTIC"
            result["note"] = "Guilloché matches the pattern this serial produces."
        elif source == "ocr":
            # An OCR'd serial we could not confirm: a non-match may be a misread
            # rather than a forgery, so stay honest and do not accuse.
            result["verdict"] = "UNVERIFIED"
            result["note"] = ("Could not auto-verify — the serial may have been "
                              "misread. Enter the serial number to verify.")
        elif score <= _TAMPER_THRESHOLD:
            result["verdict"] = "TAMPERED"
            result["note"] = ("Guilloché does not match the pattern this serial "
                              "should produce.")
        else:
            result["verdict"] = "UNVERIFIED"
            result["note"] = ("Inconclusive — pattern detail too degraded to "
                              "confirm. Upload a clearer / higher-resolution image.")
        return result
    except Exception as e:  # honest catch-all: verification must never 500
        result["note"] = f"verification error: {e}"
        return result
