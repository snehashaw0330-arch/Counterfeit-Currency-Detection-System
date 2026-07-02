"""
Phase S.3 / T2 — foreign-currency routing logic (integration layer).

Fast + isolated: exercises route_foreign without TensorFlow or real OCR by
monkeypatching the country detector / polymer analyser / per-currency
counterfeit predictor. Verifies the contract that protects the INR path and
routes any modelled currency through its counterfeit model:
  - an identified INR note never invokes the (heavy) detector;
  - only a CONFIDENT foreign detection produces a notice;
  - a currency WITH a trained model gets its verdict surfaced (BDT, AUD, …);
  - a synthetic-fake model carries its honesty caveat into the notice;
  - a currency WITHOUT a model stays identification + polymer only;
  - routing never raises.
"""

import numpy as np

import backend.foreign_routing as fr
from backend.foreign_routing import (
    route_foreign,
    serial_looks_inr,
    lacks_currency_identity,
    _FOREIGN_CONF,
)


_IMG = np.zeros((10, 10, 3), dtype=np.uint8)

_NO_MODEL = {"available": False, "currency": None, "name": None, "verdict": None,
             "confidence": None, "prob_genuine": None, "synthetic_fakes": False}


def _patch(monkeypatch, detect_result, *, polymer="PASS", counterfeit=None):
    calls = {"detect": 0, "predict": 0, "predict_ccy": None}

    def fake_detect(bgr):
        calls["detect"] += 1
        return detect_result

    def fake_predict(ccy, bgr):
        calls["predict"] += 1
        calls["predict_ccy"] = ccy
        return dict(counterfeit) if counterfeit is not None else dict(_NO_MODEL)

    monkeypatch.setattr(fr, "detect_country", fake_detect)
    monkeypatch.setattr(fr, "analyze_polymer_features",
                        lambda bgr: {"status": polymer, "details": "", "value": {}})
    monkeypatch.setattr(fr, "predict_foreign", fake_predict)
    return calls


def test_identified_inr_skips_detector(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "AUD", "confidence": 0.99, "country": "Australia"})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=True)
    assert calls["detect"] == 0 and calls["predict"] == 0
    assert cd["currency"] == "INR" and cd["method"] == "forensic_identity"
    assert notice is None and polymer is None and cf is None


def test_confident_foreign_without_model_is_identification_only(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "PHP", "confidence": 0.9, "country": "Philippines"})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert calls["predict"] == 1 and calls["predict_ccy"] == "PHP"
    assert cd["currency"] == "PHP"
    assert notice and "No counterfeit model" in notice
    assert polymer is not None
    assert cf is not None and cf["available"] is False


def test_bdt_runs_its_counterfeit_model(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "BDT", "confidence": 0.9, "country": "Bangladesh"},
                   counterfeit={"available": True, "currency": "BDT", "name": "random_forest",
                                "verdict": "FAKE", "confidence": "88.00%",
                                "prob_genuine": 0.12, "synthetic_fakes": False})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert calls["predict"] == 1 and calls["predict_ccy"] == "BDT"
    assert cf["available"] and cf["verdict"] == "FAKE"
    assert "counterfeit model was applied" in notice
    assert "synthetic" not in notice  # real physical fakes -> no caveat


def test_aud_model_verdict_and_synthetic_caveat(monkeypatch):
    _patch(monkeypatch, {"currency": "AUD", "confidence": 0.85, "country": "Australia"},
           counterfeit={"available": True, "currency": "AUD", "name": "svm_rbf",
                        "verdict": "REAL", "confidence": "95.00%",
                        "prob_genuine": 0.95, "synthetic_fakes": True})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert cf["available"] and cf["verdict"] == "REAL"
    assert "counterfeit model was applied" in notice
    assert "synthetic" in notice  # honesty caveat must surface


def test_low_confidence_foreign_is_not_routed(monkeypatch):
    calls = _patch(monkeypatch, {"currency": "GBP", "confidence": _FOREIGN_CONF - 0.05,
                                 "country": "United Kingdom"})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert calls["predict"] == 0
    assert cd["currency"] == "GBP"
    assert notice is None and polymer is None and cf is None


def test_unknown_currency_is_not_routed(monkeypatch):
    _patch(monkeypatch, {"currency": "UNKNOWN", "confidence": 0.0, "country": "Unknown"})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "UNKNOWN"
    assert notice is None and polymer is None and cf is None


def test_detected_inr_is_not_treated_as_foreign(monkeypatch):
    _patch(monkeypatch, {"currency": "INR", "confidence": 0.95, "country": "India"})
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"
    assert notice is None and polymer is None and cf is None


def test_routing_never_raises(monkeypatch):
    def boom(bgr):
        raise RuntimeError("detector blew up")
    monkeypatch.setattr(fr, "detect_country", boom)
    cd, notice, polymer, cf = route_foreign(_IMG, inr_identified=False)
    assert cd["currency"] == "INR"
    assert notice is None and polymer is None and cf is None


# --------------------------------------------------------------------------
# serial_looks_inr — the positive-INR-evidence half of the detector-skip gate.
# Regression guard for the "Canadian $20 reads cleanly → level full → detector
# skipped → INR verdict on a foreign note" bug.
# --------------------------------------------------------------------------

def test_inr_shaped_serials_accepted():
    assert serial_looks_inr("7MP 979885")    # real note
    assert serial_looks_inr("2DA*012720")    # RBI star/replacement series
    assert serial_looks_inr("000 123456")    # specimen (all-digit prefix)
    assert serial_looks_inr("?BM 979885")    # leading digit unread
    assert serial_looks_inr("  7MP 979885 ")  # whitespace tolerated


def test_foreign_or_garbage_serials_rejected():
    assert not serial_looks_inr("FTF 6743332")  # CAD: 3 letters + 7 digits
    assert not serial_looks_inr("7MP 9798857")  # 7 digits — not an INR panel
    assert not serial_looks_inr("AB1 234567")   # letter-letter-digit prefix
    assert not serial_looks_inr("7MP979885")    # no separator -> pipeline never emits
    assert not serial_looks_inr("")             # empty
    assert not serial_looks_inr(None)           # missing
    assert not serial_looks_inr(123456)         # non-string


# --------------------------------------------------------------------------
# lacks_currency_identity — the downgrade rule: a note whose currency could
# not be established (detector UNKNOWN, no INR-shaped serial) must never keep
# a REAL/SUSPICIOUS verdict. Regression guard for the "147×320 Canadian
# thumbnail read Likely Genuine" bug.
# --------------------------------------------------------------------------

_UNKNOWN_CD = {"country": "Unknown", "currency": "UNKNOWN",
               "confidence": 0.0, "method": "unknown_fallback"}
_INR_CD = {"country": "India", "currency": "INR",
           "confidence": None, "method": "forensic_identity"}


def test_unknown_currency_without_inr_serial_lacks_identity():
    assert lacks_currency_identity(_UNKNOWN_CD, None)
    assert lacks_currency_identity(_UNKNOWN_CD, "FTF 6743332")  # CAD-shaped


def test_inr_serial_is_positive_identity_even_when_detector_unknown():
    # Genuine INR note whose RBI text didn't OCR: serial shape keeps identity.
    assert not lacks_currency_identity(_UNKNOWN_CD, "7MP 979885")


def test_identified_currencies_have_identity():
    assert not lacks_currency_identity(_INR_CD, None)          # fast path
    assert not lacks_currency_identity(
        {"currency": "CAD", "country": "Canada", "confidence": 0.78}, None)


def test_lacks_identity_never_raises():
    assert lacks_currency_identity(None, None) is False
    assert lacks_currency_identity({}, None) is False
