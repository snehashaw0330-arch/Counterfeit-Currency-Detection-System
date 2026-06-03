"""
Phase I — GenAI explanation layer tests.

These exercise the DETERMINISTIC template path (no API key, no network) so they
are fast, free, and reproducible in CI. The live-Claude path shares the same
output contract; we assert the contract, not the prose.

The contract under test (backend/genai.explain):
  - always returns {summary: str, reasons: list[str], manual_checks: list[str],
    source: str}
  - never raises, even on garbage / partial / non-dict input
  - falls back to source == "template" when ANTHROPIC_API_KEY is unset
"""

import pytest

from backend import genai


@pytest.fixture(autouse=True)
def _force_template(monkeypatch):
    """Guarantee the no-key fallback for every test in this module, so we never
    touch the network or spend tokens regardless of the dev's environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _assert_explanation_shape(out):
    assert isinstance(out, dict)
    assert isinstance(out["summary"], str) and out["summary"].strip()
    assert isinstance(out["reasons"], list)
    assert all(isinstance(r, str) for r in out["reasons"])
    assert isinstance(out["manual_checks"], list)
    assert len(out["manual_checks"]) >= 1
    assert all(isinstance(m, str) for m in out["manual_checks"])
    assert out["source"] in {"template", "llm"}


def test_llm_unavailable_without_key():
    assert genai.llm_available() is False


def test_real_verdict_template():
    result = {
        "prediction": "REAL",
        "confidence": "82.00%",
        "forensic_analysis": {
            "denomination_classification": {"status": "PASS", "value": "500"},
            "structural_sanity": {"status": "PASS", "details": "ok"},
            "watermark_detection": {"status": "PASS", "details": "ok"},
        },
    }
    out = genai.explain(result)
    _assert_explanation_shape(out)
    assert out["source"] == "template"
    assert "GENUINE" in out["summary"].upper() or "REAL" in out["summary"].upper()
    # Denomination should surface in the wording.
    assert "500" in out["summary"]


def test_fake_verdict_surfaces_failed_checks():
    result = {
        "prediction": "FAKE",
        "confidence": "73.00%",
        "forensic_analysis": {
            "structural_sanity": {"status": "PASS", "details": "ok"},
            "watermark_detection": {
                "status": "FAIL",
                "details": "no Gandhi watermark gradient found",
            },
            "security_thread_detection": {
                "status": "FAIL",
                "details": "no vertical thread energy",
            },
        },
    }
    out = genai.explain(result)
    _assert_explanation_shape(out)
    assert "FAKE" in out["summary"].upper()
    # Both failed checks should drive at least one reason each.
    joined = " ".join(out["reasons"]).lower()
    assert "watermark" in joined
    assert "thread" in joined


def test_suspicious_verdict_template():
    result = {
        "prediction": "SUSPICIOUS",
        "confidence": "55.00%",
        "forensic_analysis": {
            "structural_sanity": {"status": "PASS", "details": "ok"},
        },
    }
    out = genai.explain(result)
    _assert_explanation_shape(out)
    assert "SUSPICIOUS" in out["summary"].upper() or "suspicious" in out["summary"]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "not a dict",
        123,
        [],
        {},
        {"prediction": "REAL"},  # no forensic_analysis
        {"forensic_analysis": {"x": "not a dict"}},  # malformed inner check
        {"prediction": "FAKE", "forensic_analysis": None},
    ],
)
def test_never_raises_on_garbage(bad):
    out = genai.explain(bad)
    _assert_explanation_shape(out)
    assert out["source"] == "template"


def _has_devanagari(s):
    return any("ऀ" <= ch <= "ॿ" for ch in s)


def test_hindi_template_is_devanagari():
    result = {
        "prediction": "REAL",
        "confidence": "90%",
        "forensic_analysis": {
            "denomination_classification": {"status": "PASS", "value": "100"},
        },
    }
    out = genai.explain(result, "hi")
    _assert_explanation_shape(out)
    assert out["source"] == "template"
    assert _has_devanagari(out["summary"])
    assert _has_devanagari(out["manual_checks"][0])


def test_hindi_fake_reasons_are_devanagari():
    result = {
        "prediction": "FAKE",
        "forensic_analysis": {
            "watermark_detection": {"status": "FAIL", "details": "no gradient"},
        },
    }
    out = genai.explain(result, "hi")
    assert _has_devanagari(" ".join(out["reasons"]))


@pytest.mark.parametrize("bad", [None, "x", 123])
def test_hindi_never_raises_on_garbage(bad):
    out = genai.explain(bad, "hi")
    _assert_explanation_shape(out)
    assert _has_devanagari(out["summary"])


def test_explain_endpoint_returns_explanation():
    """The /explain endpoint wraps explain() and never 500s."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    payload = {
        "prediction": "REAL",
        "confidence": "80.00%",
        "forensic_analysis": {
            "denomination_classification": {"status": "PASS", "value": "500"},
        },
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["llm_available"] is False
    _assert_explanation_shape(body["explanation"])
