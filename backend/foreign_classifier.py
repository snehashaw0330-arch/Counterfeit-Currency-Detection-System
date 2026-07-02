"""
Per-currency foreign counterfeit inference (generalises Phase T2's BDT-only
``bdt_classifier`` now that partner datasets add more currencies, e.g. AUD).

For a currency ``ccy`` it loads the best technique chosen by the trainer
(``models/<ccy>/metrics.json`` -> ``meta.best_model``) plus its scaler, and
scores an image into a genuine probability, then a 3-band REAL / SUSPICIOUS /
FAKE verdict (mirroring the INR verdict's honest middle band).

Trainers: scripts/train_bdt_counterfeit.py (BDT — JaalTaka, per-note metrics)
and scripts/train_foreign_counterfeit.py <ccy> (any other currency in the
locked ``dataset/foreign/<ccy>/`` layout).

HONESTY: ``synthetic_fakes`` is surfaced from the trainer's metrics — when the
training fakes were digitally-altered copies of the reals (not physical
counterfeits) the verdict measures manipulation-detection, and the UI must say
so rather than imply real-counterfeit power.

Graceful by design: an untrained currency returns ``{"available": False, ...}``
and the caller falls back to "identification only". Never raises. Mirrors
backend/classical.py.
"""

import glob
import json
import os

from backend.features import extract_feature_vector

_MODELS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
)

# 3-band verdict on genuine probability (honest "can't tell" middle band).
_REAL_MIN = 0.60
_FAKE_MAX = 0.40

# Per-currency cache: ccy -> {model, scaler, name, synthetic, reason}
_CACHE = {}


def _load_currency(ccy):
    """Load (once) the best model + scaler for `ccy`. Never raises."""
    entry = {"model": None, "scaler": None, "name": None,
             "synthetic": False, "reason": None}
    mdir = os.path.join(_MODELS_ROOT, ccy)
    try:
        import joblib
        available = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(mdir, "*.joblib"))
            if os.path.basename(p) != "scaler.joblib"
        )
        if not available:
            entry["reason"] = (f"no model .joblib in models/{ccy} "
                               f"(run scripts/train_foreign_counterfeit.py {ccy})")
            return entry
        name = available[0]
        metrics = os.path.join(mdir, "metrics.json")
        if os.path.exists(metrics):
            try:
                meta = json.load(open(metrics, encoding="utf-8")).get("meta", {})
                if meta.get("best_model") in available:
                    name = meta["best_model"]
                entry["synthetic"] = bool(meta.get("fakes_are_synthetic", False))
            except Exception as exc:
                entry["reason"] = f"metrics.json unreadable: {exc}"
                return entry
        scaler_path = os.path.join(mdir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            entry["reason"] = f"scaler.joblib missing in models/{ccy} (re-run the trainer)"
            return entry
        entry["model"] = joblib.load(os.path.join(mdir, f"{name}.joblib"))
        entry["scaler"] = joblib.load(scaler_path)
        entry["name"] = name
    except Exception as exc:
        entry.update(model=None, scaler=None, name=None,
                     reason=f"load failed: {exc}")
    return entry


def _get(ccy):
    ccy = str(ccy or "").lower()
    if ccy not in _CACHE:
        _CACHE[ccy] = _load_currency(ccy)
    return ccy, _CACHE[ccy]


def foreign_model_status(ccy):
    """'loaded' or the human-readable reason the model is unavailable."""
    _, e = _get(ccy)
    return "loaded" if e["model"] is not None else (e["reason"] or "not trained")


def has_foreign_model(ccy):
    _, e = _get(ccy)
    return e["model"] is not None


def _unavailable(ccy, e):
    return {"available": False, "currency": ccy.upper(), "name": e.get("name"),
            "verdict": None, "confidence": None, "prob_genuine": None,
            "synthetic_fakes": bool(e.get("synthetic", False))}


def predict_foreign(ccy, image):
    """Counterfeit verdict for a foreign note of currency `ccy`.

    Returns {available, currency, name, verdict (REAL/SUSPICIOUS/FAKE),
    confidence, prob_genuine, synthetic_fakes}. Never raises."""
    ccy, e = _get(ccy)
    if e["model"] is None or e["scaler"] is None:
        return _unavailable(ccy, e)
    try:
        vec = extract_feature_vector(image).reshape(1, -1)
        xs = e["scaler"].transform(vec)
        model = e["model"]
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            gi = classes.index(1) if 1 in classes else len(classes) - 1
            prob_gen = float(model.predict_proba(xs)[0][gi])
        else:
            prob_gen = 1.0 if int(model.predict(xs)[0]) == 1 else 0.0

        if prob_gen >= _REAL_MIN:
            verdict = "REAL"
        elif prob_gen <= _FAKE_MAX:
            verdict = "FAKE"
        else:
            verdict = "SUSPICIOUS"
        conf = prob_gen if prob_gen >= 0.5 else 1.0 - prob_gen

        return {
            "available": True,
            "currency": ccy.upper(),
            "name": e["name"],
            "verdict": verdict,
            "confidence": f"{conf * 100:.2f}%",
            "prob_genuine": round(prob_gen, 4),
            "synthetic_fakes": bool(e["synthetic"]),
        }
    except Exception:
        return _unavailable(ccy, e)
