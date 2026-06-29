"""
Phase S.3 — foreign-currency routing for /predict (integration layer).

The INR counterfeit pipeline stays the source of truth for Indian notes. This
layer is purely additive:

- When the INR-tuned pipeline already identified an Indian note, we do NOT run
  the (relatively heavy) country detector at all — the common path pays zero
  extra cost and the existing verdict is unchanged.
- Otherwise we run `country.detect_country`. If it CONFIDENTLY recognises a
  foreign currency, we attach the detected country + polymer-cue checks and the
  caller downgrades the verdict to UNVERIFIED — so an INR-tuned hard gate never
  mislabels a foreign note as FAKE. We have no counterfeit model for foreign
  currencies yet, so foreign support is honestly "detection + polymer cues".

Kept free of TensorFlow so the routing decision is unit-testable in isolation.
"""

from __future__ import annotations

from backend.country import detect_country
from backend.polymer import analyze_polymer_features
from backend.bdt_classifier import predict_bdt

# detect_country confidence required to treat a note as foreign. Codex's
# detector reaches >=0.6 only with real issuer/currency TEXT evidence (the weak
# palette-only tier caps at 0.45), so this never fires on a genuine INR note.
_FOREIGN_CONF = 0.6

_INR_MARKER = {
    "country": "India",
    "currency": "INR",
    "confidence": None,
    "method": "forensic_identity",
}


def route_foreign(bgr, inr_identified):
    """Return (country_detection, foreign_notice, polymer_features, bdt_counterfeit).

    `inr_identified` — True if the INR pipeline confidently read the note as an
    Indian note; when True the detector is skipped entirely (zero extra cost).
    `foreign_notice` is non-None only for a confidently detected foreign note.

    For Bangladesh (the one foreign currency with a real counterfeit model)
    `bdt_counterfeit` carries a REAL/SUSPICIOUS/FAKE verdict; for the other
    currencies it is None (identification + polymer cues only). Never raises."""
    try:
        if inr_identified:
            return dict(_INR_MARKER), None, None, None

        cd = detect_country(bgr)
        currency = cd.get("currency")
        confidence = cd.get("confidence") or 0.0
        if currency in ("INR", "UNKNOWN") or confidence < _FOREIGN_CONF:
            return cd, None, None, None

        polymer = analyze_polymer_features(bgr)
        country = cd.get("country", currency)

        if currency == "BDT":
            bdt = predict_bdt(bgr)
            if bdt.get("available"):
                notice = (
                    f"Detected a {country} (BDT) banknote. A Bangladesh "
                    "counterfeit model was applied — see the verdict above."
                )
            else:
                notice = (
                    f"Detected a {country} (BDT) banknote, but the counterfeit "
                    "model is unavailable (run scripts/train_bdt_counterfeit.py)."
                )
            return cd, notice, polymer, bdt

        notice = (
            f"Detected a {country} ({currency}) banknote. Counterfeit "
            "verification is tuned for Indian/Bangladeshi notes; showing currency "
            "detection and polymer security cues only."
        )
        return cd, notice, polymer, None
    except Exception:
        # Routing must never break /predict — fall back to the INR marker.
        return dict(_INR_MARKER), None, None, None
