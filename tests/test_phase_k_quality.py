"""
Phase K — readability / input-quality gate + verdict honesty.

The system must not claim a confident verdict when it couldn't actually read
the note (low-res / poorly-framed photo). These tests cover:
  - assess_readability level logic (full / partial / none)
  - the unread-labels + guidance text
  - never-raises contract
  - pipeline-level: a blank image reads nothing -> level "none"
  - the /predict endpoint always exposes the verification block and a valid
    verdict (REAL / FAKE / SUSPICIOUS / UNVERIFIED)

Level logic is unit-tested with crafted results dicts so it's deterministic and
doesn't depend on OCR succeeding on any particular fixture.
"""

import numpy as np
import pytest

from backend import forensic


_TINY = np.full((40, 90, 3), 128, np.uint8)  # image arg; level is outcome-driven


def _results(serial=None, denom=None, prop_value=None):
    return {
        "ocr_serial_number": {"status": "PASS" if serial else "FAIL",
                              "value": serial},
        "denomination_classification": {
            "status": "PASS" if denom else "INFO", "value": denom},
        "proportion_analysis": {
            "status": "PASS" if prop_value else "INFO", "value": prop_value},
    }


def test_level_full_when_serial_and_proportions():
    out = forensic.assess_readability(
        _TINY, _results(serial="7MP 979885",
                        prop_value={"deviation_pct": 1.3}))
    assert out["level"] == "full"
    assert out["serial_read"] is True
    assert out["proportions_measured"] is True
    assert out["guidance"] == ""  # nothing to retake for


def test_level_partial_when_only_denomination():
    out = forensic.assess_readability(_TINY, _results(denom="100"))
    assert out["level"] == "partial"
    assert out["serial_read"] is False
    assert out["denomination_read"] is True
    # The things the user could re-shoot for are surfaced.
    assert any("serial" in u for u in out["unread"])
    assert any("size" in u for u in out["unread"])
    assert out["guidance"]  # partial -> retake hint present


def test_level_none_when_nothing_read():
    out = forensic.assess_readability(_TINY, _results())
    assert out["level"] == "none"
    assert "retake" in out["guidance"].lower()


@pytest.mark.parametrize("bad", [None, "x", 123, {}, {"ocr_serial_number": "oops"}])
def test_never_raises(bad):
    out = forensic.assess_readability(bad, bad if isinstance(bad, dict) else {})
    assert out["level"] in {"full", "partial", "none"}
    assert isinstance(out["unread"], list)
    assert isinstance(out["guidance"], str)


def test_blank_image_reads_nothing(blank_image):
    """A flat grey image yields no identity signal -> level 'none'."""
    results = forensic.run_forensic_pipeline(blank_image)
    out = forensic.assess_readability(blank_image, results)
    assert out["level"] == "none"
    assert out["note_located"] is False


def test_predict_exposes_verification_and_valid_verdict(blank_image):
    import cv2
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    ok, buf = cv2.imencode(".jpg", blank_image)
    assert ok
    r = client.post("/predict",
                    files={"file": ("b.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["prediction"] in {"REAL", "FAKE", "SUSPICIOUS", "UNVERIFIED"}
    # A blank is never a confident REAL.
    assert body["prediction"] != "REAL"
    assert body["verification_level"] in {"full", "partial", "none"}
    v = body["verification"]
    for key in ("level", "serial_read", "proportions_measured",
                "denomination_read", "note_located", "unread", "guidance"):
        assert key in v
