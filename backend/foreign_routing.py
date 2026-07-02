"""
Phase S.3 — foreign-currency routing for /predict (integration layer).

The INR counterfeit pipeline stays the source of truth for Indian notes. This
layer is purely additive:

- When the INR-tuned pipeline already identified an Indian note, we do NOT run
  the (relatively heavy) country detector at all — the common path pays zero
  extra cost and the existing verdict is unchanged.
- Otherwise we run `country.detect_country`. If it CONFIDENTLY recognises a
  foreign currency, we attach the detected country + polymer-cue checks and try
  that currency's counterfeit model (``backend.foreign_classifier`` — trained
  per currency from ``dataset/foreign/<ccy>/``; BDT from JaalTaka, AUD from the
  partner dataset, future currencies the same way). With a model the verdict is
  its REAL/SUSPICIOUS/FAKE; without one the caller downgrades to UNVERIFIED —
  so an INR-tuned hard gate never mislabels a foreign note as FAKE.

Kept free of TensorFlow so the routing decision is unit-testable in isolation.
"""

from __future__ import annotations

import re

from backend.country import detect_country
from backend.polymer import analyze_polymer_features
from backend.foreign_classifier import predict_foreign

# Shape of an INR number-panel serial as the forensic pipeline formats it:
#   real note      "7MP 979885"   digit + 2 letters + 6 digits
#   replacement    "2DA*012720"   star separator (RBI star series)
#   specimen       "000 123456"   all-digit prefix (specimen fixtures)
#   recovered      "?BM 979885"   leading digit unread (reported as ?)
# Foreign serial formats differ (CAD = 3 letters + 7 digits, USD = letter +
# 8 digits + letter, …), so INR-shape is cheap positive evidence of an Indian
# note. Exactly 6 digits — INR panels never have 7.
_INR_SERIAL_SHAPE = re.compile(r"^(?:[?\d][A-Z]{2}|\d{3})[ *]\d{6}$")


def serial_looks_inr(serial) -> bool:
    """True if an OCR'd serial string has the Indian number-panel shape.

    Used as POSITIVE evidence for skipping the foreign-currency detector: a
    fully-readable note whose serial is NOT Indian-shaped (e.g. a Canadian
    $20 — Latin script reads cleanly, so readability alone can't tell it from
    an INR note) must still run country detection. False for None/garbage;
    never raises."""
    if not isinstance(serial, str):
        return False
    return bool(_INR_SERIAL_SHAPE.match(serial.strip()))


def lacks_currency_identity(country_detection, serial) -> bool:
    """True when a note's currency could not be established AT ALL — the
    detector ran and returned UNKNOWN, and there is no INR-shaped serial as
    positive Indian evidence.

    Such a note must never be cleared: the INR pipeline's REAL/SUSPICIOUS is
    meaningless for a note that may not be Indian (e.g. a 147×320 Canadian
    thumbnail whose text is physically below the OCR floor read "Likely
    Genuine"). The caller downgrades to UNVERIFIED — downgrade-only, so this
    can never clear a foreign note, and identified-INR notes are unaffected
    (detector answers INR, or the fast path skipped it). Never raises."""
    try:
        cd = country_detection or {}
        return cd.get("currency") == "UNKNOWN" and not serial_looks_inr(serial)
    except Exception:
        return False

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
    """Return (country_detection, foreign_notice, polymer_features, counterfeit).

    `inr_identified` — True if the INR pipeline confidently read the note as an
    Indian note; when True the detector is skipped entirely (zero extra cost).
    `foreign_notice` is non-None only for a confidently detected foreign note.

    `counterfeit` carries that currency's REAL/SUSPICIOUS/FAKE verdict when a
    trained model exists (models/<ccy>/); otherwise ``available: False`` and the
    note gets identification + polymer cues only. Never raises."""
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

        counterfeit = predict_foreign(currency, bgr)
        if counterfeit.get("available"):
            notice = (
                f"Detected a banknote from {country} ({currency}). Its "
                "counterfeit model was applied — see the verdict above."
            )
            if counterfeit.get("synthetic_fakes"):
                notice += (
                    " Note: this model was trained on synthetic (digitally-"
                    "altered) fakes, so it detects digital manipulation rather "
                    "than physical counterfeits."
                )
        else:
            notice = (
                f"Detected a banknote from {country} ({currency}). No counterfeit "
                f"model is trained for {currency} yet; showing currency "
                "identification and polymer security cues only."
            )
        return cd, notice, polymer, counterfeit
    except Exception:
        # Routing must never break /predict — fall back to the INR marker.
        return dict(_INR_MARKER), None, None, None
