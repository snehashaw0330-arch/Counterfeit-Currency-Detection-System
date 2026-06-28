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
    """Return (country_detection, foreign_notice, polymer_features).

    `inr_identified` — True if the INR pipeline confidently read the note as an
    Indian note (denomination read / full readability); when True the detector
    is skipped entirely. `foreign_notice` is non-None only for a confidently
    detected foreign note, signalling the caller to mark the verdict UNVERIFIED.
    Never raises."""
    try:
        if inr_identified:
            return dict(_INR_MARKER), None, None

        cd = detect_country(bgr)
        currency = cd.get("currency")
        confidence = cd.get("confidence") or 0.0
        if currency not in ("INR", "UNKNOWN") and confidence >= _FOREIGN_CONF:
            polymer = analyze_polymer_features(bgr)
            notice = (
                f"Detected a {cd.get('country', currency)} ({currency}) "
                "banknote. Counterfeit verification is currently tuned for "
                "Indian notes; showing currency detection and polymer security "
                "cues only."
            )
            return cd, notice, polymer
        return cd, None, None
    except Exception:
        # Routing must never break /predict — fall back to the INR marker.
        return dict(_INR_MARKER), None, None
