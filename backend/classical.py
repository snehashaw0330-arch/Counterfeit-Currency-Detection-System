"""
Phase D.5 — classical-model inference (live second opinion).

Loads the best classical technique chosen by train_classical.py
(``models/classical/metrics.json`` -> ``meta.best_model``) plus its
fitted scaler, and scores an image into a genuine probability. In
``/predict`` this runs as an INDEPENDENT SECOND OPINION alongside
the MobileNetV2 CNN and is surfaced in the response/UI for
transparency.

The combined-verdict weighting is intentionally NOT changed by this
module — recalibrating the fusion against the diagnostic harness
(with ROC/threshold selection) is Phase F (see docs/PROJECT_SCOPE.md).
Here the classical model is an additional, honest viewpoint, not a
verdict driver.

Graceful by design: if the artifacts are missing (models not yet
trained), ``predict_classical`` returns ``{"available": False, ...}``
and the rest of the pipeline is unaffected. Never raises.
"""

import json
import os

from backend.features import extract_feature_vector

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "classical",
)
_METRICS = os.path.join(_DIR, "metrics.json")

_LOADED = False
_MODEL = None
_SCALER = None
_NAME = None


def _lazy_load():
    global _LOADED, _MODEL, _SCALER, _NAME
    if _LOADED:
        return
    _LOADED = True
    try:
        import joblib
        with open(_METRICS, "r", encoding="utf-8") as fh:
            meta = json.load(fh)["meta"]
        name = meta.get("best_model")
        _MODEL = joblib.load(os.path.join(_DIR, f"{name}.joblib"))
        _SCALER = joblib.load(os.path.join(_DIR, "scaler.joblib"))
        _NAME = name
    except Exception:
        _MODEL = None
        _SCALER = None
        _NAME = None


def warmup_classical():
    """Pre-load the classical model at startup. Returns True if a
    model is available. Safe to call repeatedly; never raises."""
    _lazy_load()
    return _MODEL is not None


def classical_available():
    _lazy_load()
    return _MODEL is not None


def _unavailable(name=None):
    return {
        "available": False,
        "name": name,
        "verdict": None,
        "confidence": None,
        "prob_genuine": None,
    }


def predict_classical(image):
    """Second-opinion verdict from the best classical technique.

    Returns a dict with: available, name, verdict (REAL/FAKE),
    confidence ("xx.xx%"), prob_genuine (0..1). Never raises."""

    _lazy_load()
    if _MODEL is None or _SCALER is None:
        return _unavailable(_NAME)

    try:
        vec = extract_feature_vector(image).reshape(1, -1)
        xs = _SCALER.transform(vec)

        if hasattr(_MODEL, "predict_proba"):
            classes = list(_MODEL.classes_)
            gi = classes.index(1) if 1 in classes else len(classes) - 1
            prob_gen = float(_MODEL.predict_proba(xs)[0][gi])
        else:
            prob_gen = 1.0 if int(_MODEL.predict(xs)[0]) == 1 else 0.0

        verdict = "REAL" if prob_gen >= 0.5 else "FAKE"
        conf = prob_gen if verdict == "REAL" else 1.0 - prob_gen

        return {
            "available": True,
            "name": _NAME,
            "verdict": verdict,
            "confidence": f"{conf * 100:.2f}%",
            "prob_genuine": round(prob_gen, 4),
        }
    except Exception:
        return _unavailable(_NAME)
