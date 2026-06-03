"""
Phase I — GenAI explanation & accessibility layer.

Turns the structured counterfeit-note verdict + forensic check results into a
clear, plain-language explanation plus actionable manual-verification guidance.
This is the project's "GenAI that does something good": explainability and
accessibility (the visually-impaired use case both reference papers targeted),
NOT generation of counterfeit currency.

Design (production-grade, graceful):
  - explain(result) -> {summary, reasons, manual_checks, source}
  - When ANTHROPIC_API_KEY is set: uses Claude (Haiku — fast/cheap, this is a
    simple structured-summarization task) with a cached static system prompt
    and a JSON-schema-constrained response.
  - When the key is unset (or the SDK is missing, or the call fails): falls back
    to a DETERMINISTIC template built from the same forensic results — so the
    feature always works locally and "upgrades" to live GenAI when a key exists.
  - Never raises.

The Anthropic SDK is imported lazily so the rest of the backend has no hard
dependency on it.
"""

import os
import json

_MODEL = "claude-haiku-4-5"  # explicit: fast + cheap for per-request summaries

# Friendly names for the forensic-pipeline keys (used by both the LLM payload
# and the deterministic template).
_CHECK_NAMES = {
    "structural_sanity": "Overall structure",
    "uv_light_detection": "UV-reactive ink (visible-light proxy)",
    "watermark_detection": "Gandhi watermark",
    "ocr_serial_number": "Serial number",
    "gandhi_face_analysis": "Gandhi portrait",
    "security_thread_detection": "Security thread",
    "serial_typography_analysis": "Ascending serial size",
    "microprint_detection": "Micro-lettering / fine print",
    "hologram_detection": "Colour palette",
    "denomination_classification": "Denomination",
    "proportion_analysis": "Note proportions",
}

# Standard RBI manual checks — the honest, hardware-free things a person can do.
_MANUAL_CHECKS = [
    "Tilt the note: on ₹500/₹2000 the denomination numeral shifts colour "
    "(green↔blue) in genuine notes.",
    "Hold it to light: a genuine note shows the Mahatma Gandhi watermark and an "
    "electrotype denomination.",
    "Look for the security thread running through the note; on genuine notes it "
    "reads 'भारत' and 'RBI' and shifts colour when tilted.",
    "Feel the print: genuine notes have raised (intaglio) printing on the "
    "portrait, RBI emblem, and identification mark.",
    "Check the two serial numbers (top-left and bottom-right) match each other.",
]

# Hindi (Devanagari) equivalents so the deterministic template path is genuinely
# bilingual, not just the live-Claude path.
_CHECK_NAMES_HI = {
    "structural_sanity": "समग्र संरचना",
    "uv_light_detection": "UV-प्रतिक्रियाशील स्याही (दृश्य-प्रकाश अनुमान)",
    "watermark_detection": "गांधी वॉटरमार्क",
    "ocr_serial_number": "सीरियल नंबर",
    "gandhi_face_analysis": "गांधी चित्र",
    "security_thread_detection": "सुरक्षा धागा",
    "serial_typography_analysis": "बढ़ते आकार का सीरियल",
    "microprint_detection": "सूक्ष्म-अक्षर / बारीक छपाई",
    "hologram_detection": "रंग पैलेट",
    "denomination_classification": "मूल्यवर्ग",
    "proportion_analysis": "नोट का अनुपात",
    "bleed_line_detection": "किनारे की ब्लीड लाइनें",
    "identification_mark": "पहचान चिह्न (स्पर्श)",
}

_MANUAL_CHECKS_HI = [
    "नोट को झुकाएँ: असली ₹500/₹2000 पर अंक का रंग बदलता है (हरा↔नीला)।",
    "रोशनी के सामने देखें: असली नोट में महात्मा गांधी का वॉटरमार्क और अंक दिखता है।",
    "सुरक्षा धागा देखें; असली नोट पर वह 'भारत' और 'RBI' दिखाता है और झुकाने पर रंग बदलता है।",
    "छपाई को छूकर महसूस करें: असली नोट पर चित्र, RBI चिन्ह और पहचान चिह्न उभरे हुए होते हैं।",
    "दोनों सीरियल नंबर (ऊपर-बाएँ और नीचे-दाएँ) एक जैसे हैं या नहीं, जाँचें।",
]

_SYSTEM = (
    "You explain counterfeit-banknote screening results for Indian Rupee notes "
    "to an everyday person, accessibility-first (your text may be read aloud). "
    "You are given a system verdict (REAL / SUSPICIOUS / FAKE) and the results "
    "of several automatic security-feature checks.\n\n"
    "Rules:\n"
    "- Be clear, calm, and concrete. Short sentences. No jargon, no markdown.\n"
    "- Explain WHY the verdict was reached, citing the specific checks that "
    "passed or failed.\n"
    "- NEVER claim certainty. This is phone-photo software without special "
    "hardware; it can be wrong, especially on high-quality fakes. Say so when "
    "the verdict is FAKE or SUSPICIOUS.\n"
    "- Always include practical manual checks the person can do by hand.\n"
    "- Do NOT give any instructions for producing or improving counterfeit "
    "notes; only help verify authenticity.\n"
    "Respond using the provided JSON schema."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "manual_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "reasons", "manual_checks"],
    "additionalProperties": False,
}


def _denom(result):
    try:
        v = result["forensic_analysis"]["denomination_classification"].get("value")
        return v if isinstance(v, str) else None
    except Exception:
        return None


def _failed_checks(result, lang="en"):
    names = _CHECK_NAMES_HI if lang == "hi" else _CHECK_NAMES
    out = []
    fa = result.get("forensic_analysis", {}) or {}
    for key, check in fa.items():
        if key == "modular_ai_pipeline":
            continue
        if isinstance(check, dict) and check.get("status") == "FAIL":
            name = names.get(key, key)
            # Detail strings are English-only; omit them in Hindi to avoid a
            # jarring language mix (the check name carries the meaning).
            detail = "" if lang == "hi" else str(check.get("details", ""))[:140]
            out.append((name, detail))
    return out


def _build_user_payload(result):
    lines = []
    verdict = result.get("prediction", "UNKNOWN")
    conf = result.get("confidence", "")
    lines.append(f"Verdict: {verdict} (confidence {conf})")

    denom = _denom(result)
    lines.append(f"Denomination: {denom if denom else 'not readable'}")

    lines.append("Security checks:")
    fa = result.get("forensic_analysis", {}) or {}
    for key, check in fa.items():
        if key == "modular_ai_pipeline" or not isinstance(check, dict):
            continue
        name = _CHECK_NAMES.get(key, key)
        status = check.get("status", "?")
        detail = str(check.get("details", ""))[:120]
        lines.append(f"- {name}: {status} — {detail}")

    ml = result.get("ml_models")
    if isinstance(ml, dict):
        cnn = ml.get("cnn", {})
        clf = ml.get("classical", {})
        lines.append(
            f"ML opinions: CNN={cnn.get('verdict')} ({cnn.get('confidence')}); "
            f"classical={clf.get('verdict')} ({clf.get('confidence')}); "
            f"agreement={ml.get('agreement')}"
        )

    lines.append(
        "\nWrite: a 2-3 sentence plain-language summary of the verdict, a list "
        "of the main reasons (the checks that drove it), and the manual checks "
        "the person should do by hand."
    )
    return "\n".join(lines)


def _template_explain(result, lang="en"):
    """Deterministic explanation built directly from the forensic results.
    Used as the no-API-key fallback. Genuinely useful on its own. Bilingual."""

    verdict = result.get("prediction", "UNKNOWN")
    denom = _denom(result)
    failed = _failed_checks(result, lang)

    if lang == "hi":
        denom_str = f"₹{denom} " if denom else ""
        if verdict == "REAL":
            summary = (
                f"सिस्टम मानता है कि यह {denom_str}नोट संभवतः असली है। अधिकांश "
                f"सुरक्षा जाँच पास हुईं। यह गारंटी नहीं है — ज़रूरी हो तो हाथ से पुष्टि करें।"
            )
        elif verdict == "FAKE":
            summary = (
                f"सिस्टम ने इस {denom_str}नोट को संभवतः नकली बताया है। एक या अधिक "
                f"महत्वपूर्ण जाँच विफल रहीं। सावधानी बरतें और हाथ से जाँचें।"
            )
        elif verdict == "UNVERIFIED":
            summary = (
                f"फ़ोटो साफ़ न होने के कारण सिस्टम इस {denom_str}नोट को पढ़ नहीं सका। "
                f"कृपया बेहतर रोशनी में, नोट को फ़्रेम में भरकर दोबारा फ़ोटो लें।"
            )
        else:  # SUSPICIOUS / UNKNOWN
            summary = (
                f"सिस्टम इस {denom_str}नोट की पुष्टि नहीं कर सका। कुछ जाँच अस्पष्ट रहीं। "
                f"इसे संदिग्ध मानें और स्वीकार करने से पहले हाथ से जाँचें।"
            )
        if failed:
            reasons = [name for name, _ in failed]
        elif verdict == "REAL":
            reasons = ["सभी प्रमुख सुरक्षा जाँच पास हुईं।"]
        else:
            reasons = ["कोई एक जाँच पूरी तरह विफल नहीं हुई, पर समग्र भरोसा पर्याप्त नहीं था।"]
        return {"summary": summary, "reasons": reasons,
                "manual_checks": list(_MANUAL_CHECKS_HI)}

    denom_str = f"₹{denom} " if denom else ""
    if verdict == "REAL":
        summary = (
            f"The system thinks this {denom_str}note is likely GENUINE. Most "
            f"security checks passed. This is not a guarantee — confirm by hand "
            f"if it matters."
        )
    elif verdict == "FAKE":
        summary = (
            f"The system flagged this {denom_str}note as likely FAKE. One or "
            f"more important checks failed. Treat it with caution and verify by "
            f"hand; software on a phone photo can still be wrong."
        )
    elif verdict == "UNVERIFIED":
        summary = (
            f"The system could not read this {denom_str}note — the photo was too "
            f"unclear. Please retake it in good light with the note filling the "
            f"frame, and try again."
        )
    else:  # SUSPICIOUS / UNKNOWN
        summary = (
            f"The system could NOT confidently confirm this {denom_str}note. "
            f"Some checks were unclear. Treat it as suspicious and verify by "
            f"hand before accepting it."
        )

    if failed:
        reasons = [f"{name}: {detail}".rstrip(": ") for name, detail in failed]
    elif verdict == "REAL":
        reasons = ["All scored security checks passed."]
    else:
        reasons = [
            "No single check failed outright, but the combined confidence was "
            "not high enough to confirm the note."
        ]

    return {
        "summary": summary,
        "reasons": reasons,
        "manual_checks": list(_MANUAL_CHECKS),
    }


def llm_available():
    """True if a live Claude explanation can be produced (key present)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def explain(result, lang="en"):
    """Return {summary, reasons, manual_checks, source}. Never raises.

    lang: "en" (default) or "hi" (Hindi / Devanagari).
    source is "llm" when produced by Claude, "template" otherwise."""

    lang = "hi" if lang == "hi" else "en"
    fallback_checks = _MANUAL_CHECKS_HI if lang == "hi" else _MANUAL_CHECKS

    if not isinstance(result, dict):
        return {
            "summary": ("समझाने के लिए कोई परिणाम नहीं।" if lang == "hi"
                        else "No result to explain."),
            "reasons": [],
            "manual_checks": list(fallback_checks),
            "source": "template",
        }

    if not llm_available():
        out = _template_explain(result, lang)
        out["source"] = "template"
        return out

    try:
        import anthropic

        system_text = _SYSTEM
        if lang == "hi":
            system_text += (
                "\nIMPORTANT: Respond ENTIRELY in Hindi (Devanagari script) — "
                "summary, reasons, and manual_checks must all be in Hindi."
            )

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=700,
            system=[{
                "type": "text",
                "text": system_text,
                # Cache the static system prompt. (Won't engage below the
                # model's minimum cacheable prefix, but it's correct usage and
                # harmless — and pays off if the prompt grows.)
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _build_user_payload(result)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        return {
            "summary": str(data.get("summary", "")).strip(),
            "reasons": [str(r) for r in data.get("reasons", [])],
            "manual_checks": [str(m) for m in data.get("manual_checks", [])]
                              or list(fallback_checks),
            "source": "llm",
        }
    except Exception:
        # Any failure (no SDK, auth error, network, schema, parse) → template.
        out = _template_explain(result, lang)
        out["source"] = "template"
        return out
