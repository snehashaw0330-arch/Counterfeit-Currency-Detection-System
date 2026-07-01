import numpy as np

from backend import country


def _blank():
    return np.full((160, 320, 3), 220, dtype=np.uint8)


def test_unknown_fallback_when_no_signal(monkeypatch):
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: [])
    # Also stub the Bengali fallback so the unit test stays fast/offline (it
    # would otherwise build a real EasyOCR reader when English text is empty).
    monkeypatch.setattr(country, "_ocr_texts_bengali", lambda _img: [])
    out = country.detect_country(_blank())
    assert out["currency"] == "UNKNOWN"
    assert out["country"] == "Unknown"
    assert out["method"] == "unknown_fallback"


def test_bdt_detected_from_ocr_cues(monkeypatch):
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: ["BANGLADESH BANK 100 TAKA"])
    out = country.detect_country(_blank())
    assert out["currency"] == "BDT"
    assert out["country"] == "Bangladesh"
    assert out["method"] in {"ocr_text", "ocr_text+palette"}
    assert out["confidence"] >= 0.6


def test_bdt_detected_from_bengali_script(monkeypatch):
    """A Bangladeshi note whose issuer text is Bengali (unreadable by the
    English OCR) must still be identified as BDT via the Bengali fallback —
    this is the regression guard for the '৳1000 read as genuine ₹100' bug."""
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: [])
    monkeypatch.setattr(
        country, "_ocr_texts_bengali",
        lambda _img: ["বাংলাদেশ ব্যাংক এক হাজার টাকা"],
    )
    out = country.detect_country(_blank())
    assert out["currency"] == "BDT"
    assert out["country"] == "Bangladesh"
    assert out["confidence"] >= 0.6


def test_incidental_bengali_line_does_not_trigger_bdt(monkeypatch):
    """A short Bengali fragment (e.g. one line of the multilingual panel on an
    Indian note's reverse) must NOT reach the confident BDT tier — the signal
    is scaled by Bengali density so only a Bengali-dominant note qualifies."""
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: [])
    monkeypatch.setattr(country, "_ocr_texts_bengali", lambda _img: ["টাকা"])
    out = country.detect_country(_blank())
    assert out["currency"] != "BDT"


def test_gbp_detected_from_ocr_cues(monkeypatch):
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: ["BANK OF ENGLAND TWENTY POUNDS"])
    out = country.detect_country(_blank())
    assert out["currency"] == "GBP"
    assert out["country"] == "United Kingdom"


def test_php_detected_from_ocr_cues(monkeypatch):
    monkeypatch.setattr(
        country,
        "_ocr_text_pool",
        lambda _img: ["REPUBLIKA NG PILIPINAS BANGKO SENTRAL NG PILIPINAS"],
    )
    out = country.detect_country(_blank())
    assert out["currency"] == "PHP"
    assert out["country"] == "Philippines"


def test_palette_tiebreak_can_return_aud(monkeypatch):
    img = np.full((130, 260, 3), (160, 120, 210), dtype=np.uint8)
    monkeypatch.setattr(country, "_ocr_text_pool", lambda _img: [])
    monkeypatch.setattr(country, "_ocr_texts_bengali", lambda _img: [])
    out = country.detect_country(img)
    assert out["currency"] in {"AUD", "UNKNOWN"}
    if out["currency"] == "AUD":
        assert out["method"] == "palette_aspect_tiebreak"
        assert out["confidence"] <= 0.45
