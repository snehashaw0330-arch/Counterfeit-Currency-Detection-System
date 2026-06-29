"""
Phase S.3 / T2 — foreign-currency routing logic (integration layer).

Fast + isolated: exercises route_foreign without TensorFlow or real OCR by
monkeypatching the country detector / polymer analyser / BDT model. Verifies the
contract that protects the INR path and routes Bangladesh through the counterfeit
model:
  - an identified INR note never invokes the (heavy) detector;
  - only a CONFIDENT foreign detection produces a notice;
  - Bangladesh runs the BDT counterfeit model (verdict surfaced);
  - other foreign currencies stay identification + polymer only;
  - routing never raises.
"""

import numpy as np

import backend.foreign_routing as fr
from backend.foreign_routing import route_foreign, _FOREIGN_CONF


_IMG = np.zeros((10, 10, 3), dtype=np.uint8)


def _patch(monkeypatch, detect_result, *, polymer="PASS", bdt=None):
    calls = {"detect": 0, "bdt": 0}

    def fake_detect(bgr):
        calls["detect"] += 1
        return detect_result

    def fake_bdt(bgr):
        calls["bdt"] += 1
        return bdt if bdt is not None else {"available": False, "verdict": None}

    monkeypatch.setattr(fr, "detect_country", fake_detect)
    monkeypatch.setattr(fr, "analyze_polymer_features",
                        lambda bgr: {"status": polymer, "details": "", "value": {}})
    monkeypatch.setattr(fr, "predict_bdt", fake_bdt)
    return calls


def test_identified_inr_skips_detector(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "AUD", "confidence": 0.99, "country": "Australia"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=True)
    assert calls["detect"] == 0 and calls["bdt"] == 0
    assert cd["currency"] == "INR" and cd["method"] == "forensic_identity"
    assert notice is None and polymer is None and bdt is None


def test_confident_foreign_polymer_only(monkeypatch):
    _patch(monkeypatch, {"currency": "AUD", "confidence": 0.9, "country": "Australia"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "AUD"
    assert notice and "Australia" in notice
    assert polymer is not None and bdt is None          # no counterfeit model


def test_bdt_runs_counterfeit_model(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "BDT", "confidence": 0.9, "country": "Bangladesh"},
                   bdt={"available": True, "verdict": "FAKE", "confidence": "88.00%"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert calls["bdt"] == 1
    assert cd["currency"] == "BDT"
    assert bdt["available"] and bdt["verdict"] == "FAKE"
    assert "counterfeit model was applied" in notice


def test_bdt_model_unavailable(monkeypatch):
    _patch(monkeypatch, {"currency": "BDT", "confidence": 0.9, "country": "Bangladesh"},
           bdt={"available": False, "verdict": None})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "BDT"
    assert bdt["available"] is False
    assert "unavailable" in notice


def test_low_confidence_foreign_is_not_routed(monkeypatch):
    _patch(monkeypatch, {"currency": "GBP", "confidence": _FOREIGN_CONF - 0.05, "country": "United Kingdom"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "GBP"
    assert notice is None and polymer is None and bdt is None


def test_unknown_currency_is_not_routed(monkeypatch):
    _patch(monkeypatch, {"currency": "UNKNOWN", "confidence": 0.0, "country": "Unknown"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "UNKNOWN"
    assert notice is None and polymer is None and bdt is None


def test_detected_inr_is_not_treated_as_foreign(monkeypatch):
    _patch(monkeypatch, {"currency": "INR", "confidence": 0.95, "country": "India"})
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"
    assert notice is None and polymer is None and bdt is None


def test_routing_never_raises(monkeypatch):
    def boom(bgr):
        raise RuntimeError("detector blew up")
    monkeypatch.setattr(fr, "detect_country", boom)
    cd, notice, polymer, bdt = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"
    assert notice is None and polymer is None and bdt is None
