"""
Phase T2 — Bangladesh (BDT) counterfeit inference.

Loads the best technique chosen by scripts/train_bdt_counterfeit.py
(``models/bdt/metrics.json`` -> ``meta.best_model``) plus its scaler, and scores
an image into a genuine probability, then a 3-band REAL / SUSPICIOUS / FAKE
verdict (mirroring the INR verdict's honest middle band).

This is the ONLY foreign currency with a real counterfeit model, because
JaalTaka provides genuine + physical-counterfeit BDT images. AUD/CAD/GBP/PHP get
identification + polymer cues only (no public counterfeit data).

Graceful by design: if the model isn't trained yet, ``predict_bdt`` returns
``{"available": False, ...}`` and the caller falls back to "detection only".
Never raises. Mirrors backend/classical.py.
"""

import glob
import json
import os

from backend.features import extract_feature_vector

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "bdt",
)
_METRICS = os.path.join(_DIR, "metrics.json")

# 3-band verdict on genuine probability (honest "can't tell" middle band).
_REAL_MIN = 0.60
_FAKE_MAX = 0.40

_LOADED = False
_MODEL = None
_SCALER = None
_NAME = None
_REASON = None


def _pick_model_name():
    available = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(_DIR, "*.joblib"))
        if os.path.basename(p) != "scaler.joblib"
    )
    if not available:
        return None, f"no model .joblib in {_DIR} (run scripts/train_bdt_counterfeit.py)"
    best = None
    if os.path.exists(_METRICS):
        try:
            with open(_METRICS, "r", encoding="utf-8") as fh:
                best = json.load(fh).get("meta", {}).get("best_model")
        except Exception as exc:
            return None, f"metrics.json unreadable: {exc}"
    if best and best in available:
        return best, None
    return available[0], None


def _lazy_load():
    global _LOADED, _MODEL, _SCALER, _NAME, _REASON
    if _LOADED:
        return
    _LOADED = True
    try:
        import joblib
        name, reason = _pick_model_name()
        if name is None:
            _REASON = reason
            return
        scaler_path = os.path.join(_DIR, "scaler.joblib")
        if not os.path.exists(scaler_path):
            _REASON = "scaler.joblib missing (re-run scripts/train_bdt_counterfeit.py)"
            return
        _MODEL = joblib.load(os.path.join(_DIR, f"{name}.joblib"))
        _SCALER = joblib.load(scaler_path)
        _NAME = name
        _REASON = None
    except Exception as exc:
        _MODEL = _SCALER = _NAME = None
        _REASON = f"load failed: {exc}"


def bdt_status():
    _lazy_load()
    return "loaded" if _MODEL is not None else (_REASON or "not trained")


def warmup_bdt():
    _lazy_load()
    return _MODEL is not None


def _unavailable():
    return {"available": False, "name": _NAME, "verdict": None,
            "confidence": None, "prob_genuine": None}


def predict_bdt(image):
    """Counterfeit verdict for a Bangladesh note crop.

    Returns {available, name, verdict (REAL/SUSPICIOUS/FAKE), confidence,
    prob_genuine}. Never raises."""
    _lazy_load()
    if _MODEL is None or _SCALER is None:
        return _unavailable()
    try:
        vec = extract_feature_vector(image).reshape(1, -1)
        xs = _SCALER.transform(vec)
        if hasattr(_MODEL, "predict_proba"):
            classes = list(_MODEL.classes_)
            gi = classes.index(1) if 1 in classes else len(classes) - 1
            prob_gen = float(_MODEL.predict_proba(xs)[0][gi])
        else:
            prob_gen = 1.0 if int(_MODEL.predict(xs)[0]) == 1 else 0.0

        if prob_gen >= _REAL_MIN:
            verdict = "REAL"
        elif prob_gen <= _FAKE_MAX:
            verdict = "FAKE"
        else:
            verdict = "SUSPICIOUS"
        conf = prob_gen if prob_gen >= 0.5 else 1.0 - prob_gen

        return {
            "available": True,
            "name": _NAME,
            "verdict": verdict,
            "confidence": f"{conf * 100:.2f}%",
            "prob_genuine": round(prob_gen, 4),
        }
    except Exception:
        return _unavailable()
