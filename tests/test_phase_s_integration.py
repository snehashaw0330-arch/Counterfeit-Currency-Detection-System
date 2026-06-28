"""
Phase S.3 — foreign-currency routing logic (integration layer).

Fast + isolated: exercises route_foreign without TensorFlow or real OCR by
monkeypatching the country detector / polymer analyser. Verifies the contract
that protects the INR path:
  - an identified INR note never invokes the (heavy) detector;
  - only a CONFIDENT foreign detection produces a notice (-> caller marks
    the verdict UNVERIFIED) and polymer cues;
  - UNKNOWN / low-confidence / INR detections leave the verdict alone;
  - routing never raises.
"""

import numpy as np

import backend.foreign_routing as fr
from backend.foreign_routing import route_foreign, _FOREIGN_CONF


_IMG = np.zeros((10, 10, 3), dtype=np.uint8)


def _patch_detector(monkeypatch, result, *, polymer="PASS"):
    calls = {"n": 0}

    def fake_detect(bgr):
        calls["n"] += 1
        return result

    monkeypatch.setattr(fr, "detect_country", fake_detect)
    monkeypatch.setattr(fr, "analyze_polymer_features",
                        lambda bgr: {"status": polymer, "details": "", "value": {}})
    return calls


def test_identified_inr_skips_detector(monkeypatch):
    calls = _patch_detector(monkeypatch, {"currency": "AUD", "confidence": 0.99,
                                          "country": "Australia"})
    cd, notice, polymer = route_foreign(_IMG, inr_identified=True)
    assert calls["n"] == 0                 # detector never invoked
    assert cd["currency"] == "INR" and cd["method"] == "forensic_identity"
    assert notice is None and polymer is None


def test_confident_foreign_attaches_notice_and_polymer(monkeypatch):
    _patch_detector(monkeypatch, {"currency": "AUD", "confidence": 0.9,
                                  "country": "Australia"})
    cd, notice, polymer = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "AUD"
    assert notice and "Australia" in notice and "AUD" in notice
    assert polymer is not None and polymer["status"] == "PASS"


def test_low_confidence_foreign_is_not_routed(monkeypatch):
    _patch_detector(monkeypatch, {"currency": "GBP",
                                  "confidence": _FOREIGN_CONF - 0.05,
                                  "country": "United Kingdom"})
    cd, notice, polymer = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "GBP"
    assert notice is None and polymer is None


def test_unknown_currency_is_not_routed(monkeypatch):
    _patch_detector(monkeypatch, {"currency": "UNKNOWN", "confidence": 0.0,
                                  "country": "Unknown"})
    cd, notice, polymer = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "UNKNOWN"
    assert notice is None and polymer is None


def test_detected_inr_is_not_treated_as_foreign(monkeypatch):
    _patch_detector(monkeypatch, {"currency": "INR", "confidence": 0.95,
                                  "country": "India"})
    cd, notice, polymer = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"
    assert notice is None and polymer is None


def test_routing_never_raises(monkeypatch):
    def boom(bgr):
        raise RuntimeError("detector blew up")
    monkeypatch.setattr(fr, "detect_country", boom)
    cd, notice, polymer = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"        # safe fallback
    assert notice is None and polymer is None
