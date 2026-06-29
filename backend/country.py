"""
Country and currency detection for foreign-banknote routing.

This module is intentionally self-contained so it can be integrated by the
routing layer without disturbing the current INR-only verdict path.

Design goals
------------
- OCR-first: issuer / currency text is the strongest signal.
- Gentle heuristics: palette and aspect are only tie-breakers.
- Honest fallback: if the signal is weak or contradictory, return UNKNOWN.
- Never raises: a bad image should not break the caller.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import cv2
import numpy as np


_COUNTRY_INFO = {
    "INR": {"country": "India", "currency": "INR"},
    "BDT": {"country": "Bangladesh", "currency": "BDT"},
    "GBP": {"country": "United Kingdom", "currency": "GBP"},
    "CAD": {"country": "Canada", "currency": "CAD"},
    "AUD": {"country": "Australia", "currency": "AUD"},
    "PHP": {"country": "Philippines", "currency": "PHP"},
    "USD": {"country": "United States", "currency": "USD"},
    "EUR": {"country": "Eurozone", "currency": "EUR"},
}

_TEXT_PATTERNS = {
    "INR": [
        (r"\bRESERVE BANK OF INDIA\b", 8.0),
        (r"\bBANK OF INDIA\b", 4.0),
        (r"\bINDIA\b", 2.0),
        (r"\bRUPEES?\b", 1.5),
        (r"\bBHARATIYA\b", 2.0),
        (r"[\u0900-\u097F]", 2.5),  # Devanagari
    ],
    "BDT": [
        (r"\bBANGLADESH BANK\b", 8.0),
        (r"\bBANGLADESH\b", 3.0),
        (r"\bTAKA\b", 2.0),
        (r"[\u0980-\u09FF]", 3.0),  # Bengali script
    ],
    "GBP": [
        (r"\bBANK OF ENGLAND\b", 8.0),
        (r"\bENGLAND\b", 2.5),
        (r"\bPOUNDS?\b", 2.0),
        (r"\bSTERLING\b", 1.5),
    ],
    "CAD": [
        (r"\bBANK OF CANADA\b", 8.0),
        (r"\bBANQUE DU CANADA\b", 8.0),
        (r"\bCANADA\b", 3.0),
        (r"\bCANADE\b", 2.5),
        (r"\bBANQUE\b", 2.0),
        (r"\bGOUVERNEUR\b", 1.0),
        (r"\bDOLLARS?\b", 1.0),
    ],
    "AUD": [
        (r"\bRESERVE BANK OF AUSTRALIA\b", 8.0),
        (r"\bAUSTRALIA\b", 3.5),
        (r"\bDOLLARS?\b", 1.0),
    ],
    "PHP": [
        (r"\bREPUBLIKA NG PILIPINAS\b", 8.0),
        (r"\bBANGKO SENTRAL NG PILIPINAS\b", 8.0),
        (r"\bPILIPINAS\b", 4.0),
        (r"\bPISO\b", 2.0),
        (r"\bPESO\b", 1.5),
    ],
    "USD": [
        (r"\bFEDERAL RESERVE\b", 8.0),
        (r"\bUNITED STATES OF AMERICA\b", 8.0),
        (r"\bUNITED STATES\b", 4.0),
        (r"\bLEGAL TENDER FOR ALL DEBTS\b", 6.0),
        (r"\bLEGAL TENDER\b", 2.5),
        (r"\bTHIS NOTE\b", 1.5),
    ],
    "EUR": [
        (r"\bEUROPEAN CENTRAL BANK\b", 8.0),
        (r"\bEURO\b", 4.0),
        (r"\b(ECB|EZB|BCE)\b", 3.0),
        (r"[Α-Ω]{3,}", 1.0),  # Greek (EYPΩ) on euro notes
    ],
}

# Weak colour priors used only as tie-breakers when OCR is sparse.
# Values are HSV centroids in OpenCV ranges (H: 0-179, S/V: 0-255).
_PALETTE_PRIORS = {
    "INR": np.array([22.0, 55.0, 150.0], dtype=np.float32),
    "BDT": np.array([42.0, 80.0, 135.0], dtype=np.float32),
    "GBP": np.array([150.0, 65.0, 150.0], dtype=np.float32),
    "CAD": np.array([58.0, 80.0, 145.0], dtype=np.float32),
    "AUD": np.array([165.0, 78.0, 160.0], dtype=np.float32),
    "PHP": np.array([112.0, 75.0, 140.0], dtype=np.float32),
    "USD": np.array([60.0, 35.0, 150.0], dtype=np.float32),   # muted green-grey
    "EUR": np.array([20.0, 60.0, 150.0], dtype=np.float32),   # varies by denom; weak prior
}

_ASPECT_PRIOR = {
    "INR": 2.27,
    "BDT": 2.18,
    "GBP": 1.91,
    "CAD": 2.18,
    "AUD": 2.00,
    "PHP": 2.42,
    "USD": 2.35,
    "EUR": 1.93,
}


def _ensure_bgr(image) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image shape: {arr.shape}")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def _ocr_variants(bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    inv = 255 - thr
    up_eq = cv2.resize(eq, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up_thr = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    return [gray, eq, thr, inv, up_eq, up_thr]


def _ocr_texts_tesseract(variants: Iterable[np.ndarray]) -> list[str]:
    try:
        import pytesseract
    except Exception:
        return []

    out = []
    config = "--psm 6"
    for im in variants:
        try:
            text = pytesseract.image_to_string(im, config=config)
        except Exception:
            continue
        text = _normalize_text(text)
        if text:
            out.append(text)
    return out


def _ocr_texts_easyocr(bgr: np.ndarray) -> list[str]:
    """EasyOCR via the SHARED forensic singleton — no second Reader and no
    per-call ~3 s model load (the backend warms it at startup). Returns
    normalized text lines, or [] if EasyOCR is unavailable."""
    try:
        from backend.forensic import _get_easyocr_reader
        reader = _get_easyocr_reader()
        if reader is None:
            return []
        results = reader.readtext(bgr, detail=0, paragraph=True)
    except Exception:
        return []
    out = []
    for text in results:
        text = _normalize_text(str(text))
        if text:
            out.append(text)
    return out


def _ocr_text_pool(bgr: np.ndarray) -> list[str]:
    # EasyOCR (shared singleton) is the project's primary OCR — it replaced
    # Tesseract on banknote fonts. Tesseract is kept only as a last-resort
    # fallback if EasyOCR is unavailable (and its binary happens to be present).
    texts = _ocr_texts_easyocr(bgr)
    if not texts:
        texts.extend(_ocr_texts_tesseract(_ocr_variants(bgr)))
    # Deduplicate while preserving order.
    seen = set()
    uniq = []
    for t in texts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _text_scores(texts: Iterable[str]) -> dict[str, float]:
    scores = {code: 0.0 for code in _COUNTRY_INFO}
    joined = " ".join(texts)
    for code, patterns in _TEXT_PATTERNS.items():
        for pattern, weight in patterns:
            if re.search(pattern, joined, flags=re.IGNORECASE):
                scores[code] += weight
    return scores


def _note_hsv_signature(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if h < 4 or w < 4:
        return np.zeros(3, dtype=np.float32), 0.0
    crop = bgr[int(h * 0.08):int(h * 0.92), int(w * 0.08):int(w * 0.92)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mask = sat > 28
    pixels = hsv[mask]
    if pixels.size == 0:
        pixels = hsv.reshape(-1, 3)
    med = np.median(pixels, axis=0).astype(np.float32)
    aspect = float(max(w, h) / max(min(w, h), 1))
    return med, aspect


def _palette_scores(bgr: np.ndarray) -> dict[str, float]:
    med, aspect = _note_hsv_signature(bgr)
    scores = {code: 0.0 for code in _COUNTRY_INFO}
    if not np.any(med):
        return scores
    for code, center in _PALETTE_PRIORS.items():
        hue_delta = min(abs(float(med[0] - center[0])), 180.0 - abs(float(med[0] - center[0])))
        sat_delta = abs(float(med[1] - center[1]))
        val_delta = abs(float(med[2] - center[2]))
        dist = hue_delta * 1.7 + sat_delta * 0.08 + val_delta * 0.04
        palette = max(0.0, 2.2 - dist / 28.0)
        aspect_score = max(0.0, 1.2 - abs(aspect - _ASPECT_PRIOR[code]) * 2.5)
        scores[code] = 0.6 * palette + 0.4 * aspect_score
    return scores


def _best_country(scores: dict[str, float]) -> tuple[str, float, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_code, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return best_code, float(best_score), float(second_score)


def detect_country(image) -> dict:
    """Return {country, currency, confidence, method}.

    Confidence is a conservative 0..1 estimate. Unknown is explicit and should
    be respected by the integration layer rather than forced into a country.
    """

    unknown = {
        "country": "Unknown",
        "currency": "UNKNOWN",
        "confidence": 0.0,
        "method": "unknown_fallback",
    }

    try:
        bgr = _ensure_bgr(image)
    except Exception:
        return unknown

    try:
        texts = _ocr_text_pool(bgr)
        text_scores = _text_scores(texts)
        palette_scores = _palette_scores(bgr)

        combined = {}
        for code in _COUNTRY_INFO:
            combined[code] = text_scores[code] + 0.7 * palette_scores[code]

        best_code, best_score, second_score = _best_country(combined)
        best_text = text_scores[best_code]
        margin = best_score - second_score

        if best_text >= 6.0 and margin >= 1.0:
            confidence = min(0.98, 0.68 + 0.04 * best_text + 0.02 * margin)
            method = "ocr_text"
        elif best_text >= 3.0 and margin >= 0.8:
            confidence = min(0.88, 0.52 + 0.04 * best_text + 0.05 * margin)
            method = "ocr_text+palette"
        elif best_score >= 1.1 and margin >= 0.6:
            confidence = min(0.45, 0.18 + 0.08 * best_score)
            method = "palette_aspect_tiebreak"
        else:
            return unknown

        info = _COUNTRY_INFO[best_code]
        return {
            "country": info["country"],
            "currency": info["currency"],
            "confidence": round(float(confidence), 4),
            "method": method,
        }
    except Exception:
        return unknown
