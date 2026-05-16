"""Unit tests for Phase B's _locate_note auto-crop helper.

Tests are intentionally synthetic so they don't depend on EasyOCR or
the model file — they only exercise the contour detection and
perspective-rectify code paths in backend/forensic.py.
"""

import cv2
import numpy as np
import pytest

from backend import forensic


# --------------------------------------------------------------------- helpers


def _synthetic_note_on_background(
    bg_size: tuple[int, int] = (1000, 1400),
    note_size: tuple[int, int] = (400, 880),  # ~25% of frame, banknote aspect 2.2
    note_position: tuple[int, int] = (300, 260),
) -> np.ndarray:
    """Render a brown banknote rectangle on a uniform tan background.

    bg_size and note_size are (height, width). The note is drawn as a
    filled rectangle with a darker border so Canny can find it."""

    bg_h, bg_w = bg_size
    note_h, note_w = note_size
    y0, x0 = note_position

    img = np.full((bg_h, bg_w, 3), (200, 220, 230), dtype=np.uint8)

    note_color = (140, 165, 180)
    border_color = (40, 50, 60)

    cv2.rectangle(
        img, (x0, y0), (x0 + note_w, y0 + note_h),
        note_color, thickness=-1,
    )
    cv2.rectangle(
        img, (x0, y0), (x0 + note_w, y0 + note_h),
        border_color, thickness=4,
    )

    # Throw in some interior structure so the contour finder doesn't
    # see the note as featureless.
    cv2.line(
        img,
        (x0 + 20, y0 + note_h // 2),
        (x0 + note_w - 20, y0 + note_h // 2),
        (90, 110, 130), 2,
    )

    return img


# --------------------------------------------------------------------- tests


def test_locate_note_crops_smaller_than_background():
    """When fed a note on a larger background, the helper must return
    a tighter image whose area is meaningfully smaller than the input
    and whose aspect approximates a banknote."""

    img = _synthetic_note_on_background()
    h0, w0 = img.shape[:2]

    out = forensic._locate_note(img)
    h1, w1 = out.shape[:2]

    assert (h1 * w1) < (h0 * w0) * 0.75, (
        f"_locate_note did not crop: in={h0}x{w0} ({h0*w0}), "
        f"out={h1}x{w1} ({h1*w1})"
    )

    aspect = w1 / max(h1, 1)
    assert 1.4 <= aspect <= 3.0, (
        f"cropped aspect {aspect:.2f} is not banknote-shaped"
    )


def test_locate_note_returns_unchanged_when_input_is_already_a_note():
    """A pure-note image with no surrounding background has no quad to
    crop to — the helper must return the input unchanged rather than
    over-aggressively shrinking it."""

    # Simulate a pre-cropped note: just a banknote-aspect rectangle
    # filling the whole frame with thin organic texture (lines, dots)
    # — but no inner rectangle contour that would look like another
    # candidate note quad.
    img = np.full((400, 880, 3), (140, 165, 180), dtype=np.uint8)
    cv2.line(img, (40, 200), (840, 200), (90, 110, 130), 2)
    for x in range(80, 800, 40):
        cv2.circle(img, (x, 100), 6, (60, 80, 100), -1)

    out = forensic._locate_note(img)

    # Allow tiny cropping (within 10%) but the helper should not
    # mangle a frame that's already a note.
    h0, w0 = img.shape[:2]
    h1, w1 = out.shape[:2]
    assert h1 * w1 >= h0 * w0 * 0.9, (
        f"helper over-cropped a clean note: {h0}x{w0} -> {h1}x{w1}"
    )


@pytest.mark.parametrize("img", [
    np.zeros((400, 600, 3), dtype=np.uint8),                       # blank
    np.full((400, 600, 3), 255, dtype=np.uint8),                   # white
    np.random.RandomState(0).randint(                              # noise
        0, 255, (400, 600, 3), dtype=np.uint8
    ),
])
def test_locate_note_safe_fallback_on_garbage(img):
    """Blank, white, or pure-noise inputs have no detectable quad.
    The helper must NOT raise and must return a usable BGR image
    (either the original or some safe pass-through)."""

    out = forensic._locate_note(img)

    assert out is not None
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.dtype == np.uint8
