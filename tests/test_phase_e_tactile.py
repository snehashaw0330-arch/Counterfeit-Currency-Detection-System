"""
Phase E — tactile/edge security features: bleed lines + identification mark.

These are honest checks: PASS only on positive evidence at adequate resolution,
INFO otherwise, and they NEVER FAIL (a miscount on a phone photo is a resolution
problem, not proof of forgery — same rule as micro-print / the rejected ORB
motif check). Tests assert that contract + never-raises + denomination gating.
"""

import numpy as np
import pytest

from backend import forensic


# A mid-size synthetic note image (BGR). Content is irrelevant for the contract
# tests; only resolution + denomination gating matter.
_IMG = np.full((300, 700, 3), 200, np.uint8)
_SMALL = np.full((80, 180, 3), 200, np.uint8)


# ---------------- bleed lines ----------------

def test_bleed_lines_unknown_denomination_is_info():
    out = forensic.analyze_bleed_lines(_IMG, denomination=None)
    assert out["status"] == "INFO"
    out = forensic.analyze_bleed_lines(_IMG, denomination="10")  # no spec
    assert out["status"] == "INFO"


def test_bleed_lines_low_res_is_info_not_fail():
    out = forensic.analyze_bleed_lines(_SMALL, denomination="500")
    assert out["status"] == "INFO"
    assert "low-resolution" in out["details"].lower()


@pytest.mark.parametrize("denom", ["100", "200", "500", "2000"])
def test_bleed_lines_never_fails(denom):
    # Across plain / noisy / striped inputs the status is only ever PASS or INFO.
    rng = np.random.default_rng(0)
    for img in (
        np.full((1000, 2000, 3), 180, np.uint8),
        rng.integers(0, 255, (1000, 2000, 3), dtype=np.uint8),
    ):
        out = forensic.analyze_bleed_lines(img, denomination=denom)
        assert out["status"] in {"PASS", "INFO"}


def test_bleed_lines_never_raises():
    for bad in (None, np.zeros((1, 1, 3), np.uint8)):
        out = forensic.analyze_bleed_lines(bad, denomination="500")
        assert out["status"] in {"PASS", "INFO"}


# ---------------- identification mark ----------------

def test_id_mark_unknown_denomination_is_info():
    out = forensic.analyze_identification_mark(_IMG, denomination=None)
    assert out["status"] == "INFO"


@pytest.mark.parametrize("denom,shape", [
    ("100", "triangle"), ("500", "circle"), ("2000", "rectangle"),
])
def test_id_mark_mentions_expected_shape_and_never_fails(denom, shape):
    out = forensic.analyze_identification_mark(_IMG, denomination=denom)
    assert out["status"] in {"PASS", "INFO"}
    assert shape in out["details"].lower()


def test_id_mark_never_raises():
    for bad in (None, np.zeros((1, 1, 3), np.uint8)):
        out = forensic.analyze_identification_mark(bad, denomination="100")
        assert out["status"] in {"PASS", "INFO"}


# ---------------- pipeline wiring ----------------

def test_pipeline_includes_new_keys(synthetic_note):
    res = forensic.run_forensic_pipeline(synthetic_note)
    for key in ("bleed_line_detection", "identification_mark"):
        assert key in res
        assert res[key]["status"] in {"PASS", "FAIL", "INFO"}
