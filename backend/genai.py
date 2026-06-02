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


def _failed_checks(result):
    out = []
    fa = result.get("forensic_analysis", {}) or {}
    for key, check in fa.items():
        if key == "modular_ai_pipeline":
            continue
        if isinstance(check, dict) and check.get("status") == "FAIL":
            name = _CHECK_NAMES.get(key, key)
            detail = str(check.get("details", ""))[:140]
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


def _template_explain(result):
    """Deterministic explanation built directly from the forensic results.
    Used as the no-API-key fallback. Genuinely useful on its own."""

    verdict = result.get("prediction", "UNKNOWN")
    denom = _denom(result)
    denom_str = f"₹{denom} " if denom else ""
    failed = _failed_checks(result)

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
    else:  # SUSPICIOUS / UNKNOWN
        summary = (
            f"The system could NOT confidently confirm this {denom_str}note. "
            f"Some checks were unclear. Treat it as suspicious and verify by "
            f"hand before accepting it."
        )

    if failed:
        reasons = [f"{name}: {detail}" for name, detail in failed]
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


def explain(result):
    """Return {summary, reasons, manual_checks, source}. Never raises.

    source is "llm" when produced by Claude, "template" otherwise."""

    if not isinstance(result, dict):
        return {
            "summary": "No result to explain.",
            "reasons": [],
            "manual_checks": list(_MANUAL_CHECKS),
            "source": "template",
        }

    if not llm_available():
        out = _template_explain(result)
        out["source"] = "template"
        return out

    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=700,
            system=[{
                "type": "text",
                "text": _SYSTEM,
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
                              or list(_MANUAL_CHECKS),
            "source": "llm",
        }
    except Exception:
        # Any failure (no SDK, auth error, network, schema, parse) → template.
        out = _template_explain(result)
        out["source"] = "template"
        return out
