"""
Phase P.1 — digital-tamper / Error-Level-Analysis check.

ELA is a heuristic, so the check is INFO-only (never drives the verdict). Tests
assert the contract: always INFO, sensible details, never raises, wired into the
pipeline.
"""

import numpy as np
import pytest

from backend import forensic


def test_uniform_photo_reads_uniform():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (400, 700, 3), dtype=np.uint8)  # photo-like noise
    out = forensic.analyze_tamper_ela(img)
    assert out["status"] == "INFO"
    assert "uniform" in out["details"].lower() or "error" in out["details"].lower()


def test_always_info_never_fails():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (500, 900, 3), dtype=np.uint8)
    # paste a hard-edged bright block (splice-like)
    img[200:260, 400:480] = 255
    out = forensic.analyze_tamper_ela(img)
    assert out["status"] == "INFO"  # heuristic — informational, never FAIL


def test_too_small_is_info():
    out = forensic.analyze_tamper_ela(np.zeros((20, 20, 3), np.uint8))
    assert out["status"] == "INFO"
    assert "small" in out["details"].lower()


@pytest.mark.parametrize("bad", [None, np.zeros((1, 1, 3), np.uint8)])
def test_never_raises(bad):
    out = forensic.analyze_tamper_ela(bad)
    assert out["status"] == "INFO"


def test_pipeline_includes_tamper_key(synthetic_note):
    res = forensic.run_forensic_pipeline(synthetic_note)
    assert "tamper_detection" in res
    assert res["tamper_detection"]["status"] in {"PASS", "FAIL", "INFO"}
