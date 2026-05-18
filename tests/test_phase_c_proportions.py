"""Phase C-1 acceptance tests for the banknote proportion check.

The proportion check compares the detected note quad's aspect
ratio against the canonical RBI aspect for the OCR'd
denomination and flags clear deviation (digital stretching,
wrong-size paper).
"""

import cv2
import numpy as np
import pytest

from backend import forensic


# --------------------------------------------------------------------- helpers


def _note_on_bg(
    note_w: int,
    note_h: int,
    bg_pad: int = 200,
) -> np.ndarray:
    """Render a banknote-coloured rectangle on a tan background.

    The note has a dark border so Canny + contour can find a
    clean quad. note_w / note_h directly control the quad's
    aspect, so a test can dial in any desired ratio."""

    bg_w = note_w + 2 * bg_pad
    bg_h = note_h + 2 * bg_pad
    img = np.full((bg_h, bg_w, 3), (200, 220, 230), dtype=np.uint8)

    note_color = (140, 165, 180)
    border_color = (40, 50, 60)
    x0, y0 = bg_pad, bg_pad

    cv2.rectangle(
        img, (x0, y0), (x0 + note_w, y0 + note_h),
        note_color, thickness=-1,
    )
    cv2.rectangle(
        img, (x0, y0), (x0 + note_w, y0 + note_h),
        border_color, thickness=4,
    )
    cv2.line(
        img,
        (x0 + 20, y0 + note_h // 2),
        (x0 + note_w - 20, y0 + note_h // 2),
        (90, 110, 130), 2,
    )
    return img


# --------------------------------------------------------------------- tests


def test_pass_on_canonical_rs50_proportions():
    """A note rendered at the exact Rs 50 RBI aspect (2.045)
    must PASS the proportion check."""

    # 135 mm × 66 mm → 2.045 — render at 10 px / mm
    img = _note_on_bg(note_w=1350, note_h=660)

    result = forensic.analyze_proportions(img, denomination="50")

    assert result["status"] == "PASS", result
    val = result["value"]
    assert abs(val["expected_aspect"] - 2.045) < 0.01
    assert val["deviation_pct"] < 2.0  # quad detection has tiny rounding


def test_fail_on_30pct_horizontal_stretch():
    """A note stretched 30% horizontally has aspect ~2.66 (instead
    of 2.045 for Rs 50). Deviation ~30%; must FAIL with the right
    numbers in the value payload."""

    # Width 30% larger than canonical, same height
    img = _note_on_bg(note_w=1755, note_h=660)  # 135*1.3=175.5, ×10

    result = forensic.analyze_proportions(img, denomination="50")

    assert result["status"] == "FAIL", result
    val = result["value"]
    # Actual ~2.66; expected 2.045; deviation ~30%
    assert val["actual_aspect"] > 2.5
    assert val["deviation_pct"] > 20.0
    assert "deviation" in result["details"]


def test_info_when_no_quad_detectable():
    """A blank image has no detectable quad — the check returns
    INFO (absence of signal, not proof of fakery)."""

    blank = np.full((400, 800, 3), 220, dtype=np.uint8)
    result = forensic.analyze_proportions(blank, denomination="50")

    assert result["status"] == "INFO"
    assert result["value"] is None


def test_info_when_denomination_unknown():
    """Without a denomination we have no canonical aspect to
    compare against — return INFO, never FAIL."""

    img = _note_on_bg(note_w=1350, note_h=660)

    # None
    result = forensic.analyze_proportions(img, denomination=None)
    assert result["status"] == "INFO"
    assert result["value"] is None

    # Garbage denom
    result = forensic.analyze_proportions(img, denomination="999")
    assert result["status"] == "INFO"


def test_rs2000_canonical_aspect():
    """Sanity-check a different denomination's canonical aspect
    (166/66 = 2.515)."""

    img = _note_on_bg(note_w=1660, note_h=660)
    result = forensic.analyze_proportions(img, denomination="2000")

    assert result["status"] == "PASS", result
    assert abs(result["value"]["expected_aspect"] - 2.515) < 0.01


@pytest.mark.parametrize("denom,w_mm,h_mm", [
    ("10",   123, 63),
    ("20",   129, 63),
    ("50",   135, 66),
    ("100",  142, 66),
    ("200",  146, 66),
    ("500",  150, 66),
    ("2000", 166, 66),
])
def test_every_denomination_passes_at_canonical_size(denom, w_mm, h_mm):
    """For every supported denomination, a synthetic note at the
    canonical mm aspect must PASS the proportion check."""

    img = _note_on_bg(note_w=w_mm * 10, note_h=h_mm * 10)
    result = forensic.analyze_proportions(img, denomination=denom)

    assert result["status"] == "PASS", (denom, result)
    assert result["value"]["deviation_pct"] < 5.0


def test_proportion_analysis_present_in_pipeline_output():
    """The pipeline orchestrator must surface proportion_analysis
    as a top-level key with the standard shape."""

    img = _note_on_bg(note_w=1350, note_h=660)
    result = forensic.run_forensic_pipeline(img)

    assert "proportion_analysis" in result
    pa = result["proportion_analysis"]
    assert pa["status"] in {"PASS", "FAIL", "INFO"}
    assert "details" in pa


# --------------------------------------------------------------------- regression on the real stretched fixture


def test_stretched_fixture_flagged_as_fake():
    """The committed digital_stretch fixture
    (real/real_100_obv.png stretched 30% horizontally) must
    FAIL the proportion check with a large deviation. This
    is the end-to-end regression for the feature on a real
    image, not a synthetic in-memory note."""

    import pathlib
    import cv2 as _cv

    path = (
        pathlib.Path(__file__).parent
        / "sample_notes" / "fake" / "fake_11_stretched_100.jpg"
    )
    img = _cv.imread(str(path))
    assert img is not None, f"fixture missing: {path}"

    result = forensic.analyze_proportions(img, denomination="100")

    assert result["status"] == "FAIL", result
    val = result["value"]
    assert val is not None
    assert val["expected_aspect"] == pytest.approx(2.152, abs=0.01)
    assert val["deviation_pct"] > 20.0, (
        f"stretched fixture should show >20% deviation, got "
        f"{val['deviation_pct']}%"
    )
