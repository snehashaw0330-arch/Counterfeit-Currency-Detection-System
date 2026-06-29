"""
Phase R.3 — digital PUF (enroll / verify).

Fast + isolated: each test uses a temp registry (monkeypatched path) and
deterministic textured images (the serial-seeded guilloché tokens) as stand-in
notes, so no model/OCR load and no pollution of the real registry.
"""

import cv2
import numpy as np
import pytest

import backend.puf as puf
from backend.puf import (
    compute_fingerprint,
    hamming_distance,
    enroll,
    verify,
    _MATCH_MAX_DISTANCE,
)
from backend.security_pattern import secure_note_png


def _img(serial):
    png = secure_note_png(serial, size=500)
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(puf, "_REGISTRY_PATH", str(tmp_path / "puf_registry.json"))
    yield


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_deterministic():
    img = _img("NOTE-A")
    assert compute_fingerprint(img) == compute_fingerprint(img)


def test_fingerprint_distinguishes_notes():
    assert hamming_distance(compute_fingerprint(_img("NOTE-A")),
                            compute_fingerprint(_img("NOTE-B"))) > _MATCH_MAX_DISTANCE


def test_fingerprint_never_raises_on_garbage():
    assert isinstance(compute_fingerprint(np.zeros((3, 3, 3), np.uint8)), str)


# ---------------------------------------------------------------------------
# Enroll / verify
# ---------------------------------------------------------------------------

def test_enroll_then_verify_same_is_authentic(registry):
    img = _img("NOTE-A")
    e = enroll(img, "noteA")
    assert e["status"] == "ENROLLED"
    r = verify(img, "noteA")
    assert r["verdict"] == "AUTHENTIC"
    assert r["distance"] <= _MATCH_MAX_DISTANCE


def test_verify_survives_jpeg(registry):
    img = _img("NOTE-A")
    enroll(img, "noteA")
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    r = verify(cv2.imdecode(enc, cv2.IMREAD_COLOR), "noteA")
    assert r["verdict"] == "AUTHENTIC"


def test_verify_different_note_no_match(registry):
    enroll(_img("NOTE-A"), "noteA")
    r = verify(_img("NOTE-B"), "noteA")
    assert r["verdict"] == "NO_MATCH"
    assert r["distance"] > _MATCH_MAX_DISTANCE


def test_verify_unknown_id(registry):
    enroll(_img("NOTE-A"), "noteA")
    r = verify(_img("NOTE-A"), "does-not-exist")
    assert r["verdict"] == "UNKNOWN"


def test_verify_empty_registry(registry):
    r = verify(_img("NOTE-A"), "noteA")
    assert r["verdict"] == "UNKNOWN"
    assert "empty" in r["note"].lower()


def test_verify_without_id_finds_best_match(registry):
    enroll(_img("NOTE-A"), "noteA")
    enroll(_img("NOTE-B"), "noteB")
    r = verify(_img("NOTE-B"), None)
    assert r["verdict"] == "AUTHENTIC"
    assert r["note_id"] == "noteB"


def test_reenroll_overwrites(registry):
    enroll(_img("NOTE-A"), "noteA")
    e = enroll(_img("NOTE-A"), "noteA")
    assert e["status"] == "REENROLLED"


def test_enroll_requires_note_id(registry):
    assert enroll(_img("NOTE-A"), "  ")["status"] == "ERROR"


def test_verify_never_raises_on_garbage(registry):
    enroll(_img("NOTE-A"), "noteA")
    r = verify(np.random.randint(0, 255, (20, 20, 3), np.uint8), "noteA")
    assert r["verdict"] in {"AUTHENTIC", "NO_MATCH", "UNKNOWN"}
