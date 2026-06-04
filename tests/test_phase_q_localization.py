"""Unit tests for the Track-A robust localization fallback.

`_detect_note_quad` is now a two-stage detector:

  1. `_detect_note_quad_edges` — the original Canny/contour + rotated-rect
     path (fast, clean notes on a plain background).
  2. `_segment_note_quad` — a GrabCut foreground segmentation seeded by
     spectral-residual saliency, used only when (1) finds nothing. This is
     what recovers a note in a cluttered / low-light frame (the camera
     ₹100→₹50 misread root cause).

These tests are synthetic so they don't need EasyOCR or the model file —
they exercise only the OpenCV localization code in backend/forensic.py.
The change is strictly additive: the edge path is unchanged, so clean-note
behavior must be identical and the new path only adds recoveries.
"""

import cv2
import numpy as np
import pytest

from backend import forensic


# --------------------------------------------------------------------- helpers


def _cluttered_soft_note(
    blur: int = 9,
    clutter: int = 30,
    seed: int = 7,
) -> np.ndarray:
    """A low-contrast banknote on a busy, noisy desk, lightly blurred so the
    note's border does NOT form a clean Canny edge — the exact failure the
    edge detector can't handle but region segmentation can.

    With these defaults the edge detector returns None while the segmentation
    fallback recovers a banknote-aspect quad (verified empirically)."""

    rng = np.random.RandomState(seed)
    bg = rng.randint(60, 110, (900, 1300, 3), dtype=np.uint8)
    for _ in range(clutter):
        x, y = rng.randint(0, 1250), rng.randint(0, 870)
        cv2.rectangle(
            bg, (x, y),
            (x + rng.randint(15, 90), y + rng.randint(8, 50)),
            tuple(int(c) for c in rng.randint(40, 160, 3)), -1,
        )
    note = np.full((300, 640, 3), (120, 150, 120), dtype=np.uint8)
    cv2.putText(note, "500", (60, 205), cv2.FONT_HERSHEY_SIMPLEX, 4,
                (80, 100, 80), 9)
    bg[300:600, 330:970] = note
    if blur:
        bg = cv2.GaussianBlur(bg, (blur, blur), 0)
    return bg


# --------------------------------------------------------------------- tests


def test_edge_path_unchanged_finds_clean_note():
    """The fast edge path still localizes a clean, sharp-edged note — the
    refactor must not regress the common case."""

    # Landscape note (880×400, aspect 2.2) well inside a 1400×1000 frame.
    bg = np.full((1000, 1400, 3), (205, 220, 230), dtype=np.uint8)
    cv2.rectangle(bg, (260, 300), (1140, 700), (140, 165, 180), -1)
    cv2.rectangle(bg, (260, 300), (1140, 700), (40, 50, 60), 4)

    det = forensic._detect_note_quad_edges(bg)
    assert det is not None, "edge detector regressed on a clean note"
    assert forensic._BANKNOTE_ASPECT_LO <= det["aspect"] <= forensic._BANKNOTE_ASPECT_HI


def test_segmentation_recovers_note_where_edges_fail():
    """The headline case: a soft-edged note in clutter. Edge detection finds
    nothing; the GrabCut fallback recovers a banknote-aspect quad; and the
    combined detector therefore returns a result (not None)."""

    img = _cluttered_soft_note()

    assert forensic._detect_note_quad_edges(img) is None, (
        "scene was supposed to defeat the edge detector — retune the fixture"
    )

    seg = forensic._segment_note_quad(img)
    assert seg is not None, "segmentation failed to recover the note"
    assert forensic._BANKNOTE_ASPECT_LO <= seg["aspect"] <= forensic._BANKNOTE_ASPECT_HI

    combined = forensic._detect_note_quad(img)
    assert combined is not None, "orchestrator did not fall back to segmentation"


def test_locate_note_crops_via_segmentation_fallback():
    """End-to-end: the segmentation recovery flows through `_locate_note`, so a
    note the edge path missed now yields a tighter, banknote-aspect crop
    instead of the full cluttered frame."""

    img = _cluttered_soft_note()
    h0, w0 = img.shape[:2]

    out = forensic._locate_note(img)
    h1, w1 = out.shape[:2]

    assert (h1 * w1) < (h0 * w0) * 0.85, "fallback crop was not tighter"
    aspect = w1 / max(h1, 1)
    assert 1.4 <= aspect <= 3.0, f"recovered crop aspect {aspect:.2f} not banknote-shaped"


@pytest.mark.parametrize("img", [
    np.zeros((400, 600, 3), dtype=np.uint8),                       # blank
    np.full((400, 600, 3), 255, dtype=np.uint8),                   # white
    np.random.RandomState(0).randint(                              # noise
        0, 255, (400, 600, 3), dtype=np.uint8
    ),
])
def test_segmentation_safe_on_garbage(img):
    """Blank / white / pure-noise frames have no banknote-shaped region. The
    fallback must NOT hallucinate a quad in them — the aspect + fill gates
    must reject, returning None (so we fall through to the unchanged image)."""

    assert forensic._segment_note_quad(img) is None


def test_detect_note_quad_memoized_per_frame():
    """The content-keyed memo returns the SAME object for repeated calls on an
    identical frame (so the ~5 callers per /predict don't each re-run GrabCut),
    and distinct frames get distinct cache entries."""

    forensic._QUAD_CACHE.clear()
    img = _cluttered_soft_note()

    first = forensic._detect_note_quad(img)
    second = forensic._detect_note_quad(img.copy())  # equal content, new array
    assert first is second, "memo did not collapse identical frames"
    assert len(forensic._QUAD_CACHE) == 1

    other = _cluttered_soft_note(seed=11)
    forensic._detect_note_quad(other)
    assert len(forensic._QUAD_CACHE) == 2


def test_quad_cache_bounded():
    """The memo is a bounded LRU — it never grows without limit."""

    forensic._QUAD_CACHE.clear()
    for i in range(forensic._QUAD_CACHE_MAX + 5):
        forensic._detect_note_quad(_cluttered_soft_note(seed=100 + i))
    assert len(forensic._QUAD_CACHE) <= forensic._QUAD_CACHE_MAX
