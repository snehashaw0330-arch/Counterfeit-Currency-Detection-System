"""Phase A acceptance tests for the EasyOCR-driven serial + denomination paths.

The whole suite is manifest-driven: every assertion reads ground truth from
`tests/sample_notes/manifest.json`. Adding a new fixture means dropping the
image in the right folder and appending one manifest entry — no test edits
required.

Acceptance gate (from docs/IMPLEMENTATION_PLAN.md, Phase A):

- Phone-photo grade: >= 4 of 6 known-serial phone samples must produce the
  exact serial string. The blurry derivative and the full-resolution
  handheld-with-sky-background sample are the two hardest cases and may
  fail — Phase B (auto-crop) will lift the handheld one specifically.
- Phone-photo grade: at most 1 of 8 phone samples may misclassify
  denomination.
- Clean-grade obverse: at most 1 of 5 may misclassify denomination.
- Clean-grade obverse: at most 1 of 5 may return a format-invalid serial.
- Fake/structural grade: 0 of the 4 structural fakes (blank, noise, half-
  cropped) may produce a non-null serial. Content-degraded fakes (inverted,
  desaturated, blurred-real, side-by-side) are excluded — they contain
  actual banknote pixels and OCR can legitimately read text from them.
"""

import json
import pathlib
import re

import cv2
import pytest

from backend.forensic import (
    classify_denomination,
    extract_serial_number,
    warmup_ocr,
)


# --------------------------------------------------------------------- helpers


SAMPLE_ROOT = pathlib.Path(__file__).parent / "sample_notes"
MANIFEST = SAMPLE_ROOT / "manifest.json"

SERIAL_FORMAT_RE = re.compile(r"^[A-Z0-9?]{3} [0-9]{6,7}$")


@pytest.fixture(scope="module", autouse=True)
def _warmup():
    warmup_ocr()


def _load_manifest():
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _read(path_key: str):
    img = cv2.imread(str(SAMPLE_ROOT / path_key))
    assert img is not None, f"fixture missing on disk: {path_key}"
    return img


def _by(
    grade: str,
    *,
    has_serial: bool | None = None,
    side: str | None = None,
    fake_type: str | None = None,
):
    out = {}
    for k, v in _load_manifest().items():
        if v["grade"] != grade:
            continue
        if has_serial is True and v["serial"] is None:
            continue
        if has_serial is False and v["serial"] is not None:
            continue
        if side is not None and v["side"] != side:
            continue
        if fake_type is not None and v.get("fake_type") != fake_type:
            continue
        out[k] = v
    return out


# --------------------------------------------------------------------- phone tests


def test_phone_serial_exact_match():
    """Known-serial phone samples must produce the exact ground-truth string.

    Tolerance: up to ~33% may fail. The two expected hard cases:
      - The synthetic blurry derivative (9px motion blur kills serials)
      - The full-resolution handheld sample with sky background and tilt
        (the recogniser confuses "4AA" for "444" on the tilted angle).
    Phase B auto-crop will recover the handheld case."""
    samples = _by("phone", has_serial=True)
    assert len(samples) >= 4, "manifest regression: lost phone known-serial fixtures"

    misses = []
    for path_key, meta in samples.items():
        result = extract_serial_number(_read(path_key))
        got = result.get("value")
        if got != meta["serial"]:
            misses.append((path_key, meta["serial"], got))

    # ~33% tolerance, rounded down, min 1
    allowed = max(1, len(samples) // 3)
    assert len(misses) <= allowed, (
        f"{len(misses)} phone serials mismatched "
        f"(allowed up to {allowed}): {misses}"
    )


def test_phone_denom_exact_match():
    """Every phone sample must classify to its ground-truth denomination.

    Tolerance: 1 failure allowed."""
    samples = _by("phone")
    assert len(samples) >= 5, "manifest regression: lost phone fixtures"

    misses = []
    for path_key, meta in samples.items():
        result = classify_denomination(_read(path_key))
        got = result.get("value")
        if str(got) != str(meta["denom"]):
            misses.append((path_key, meta["denom"], got))

    allowed = max(1, len(samples) // 5)
    assert len(misses) <= allowed, (
        f"{len(misses)} phone denominations mismatched "
        f"(allowed up to {allowed}): {misses}"
    )


# --------------------------------------------------------------------- clean tests


def test_clean_obverse_denom_match():
    """Clean Wikipedia obverse samples must classify correctly."""
    samples = _by("clean", side="obverse")
    assert len(samples) >= 5, "manifest regression: lost clean obverse fixtures"

    misses = []
    for path_key, meta in samples.items():
        result = classify_denomination(_read(path_key))
        got = result.get("value")
        if str(got) != str(meta["denom"]):
            misses.append((path_key, meta["denom"], got))

    allowed = max(1, len(samples) // 5)
    assert len(misses) <= allowed, (
        f"{len(misses)} clean obverse denominations mismatched "
        f"(allowed up to {allowed}): {misses}"
    )


def test_clean_obverse_serial_format():
    """Clean Wikipedia obverse samples must return a format-valid serial.

    We don't know the ground-truth digits on the Wikipedia images, only
    the layout — so we assert on the regex pattern."""
    samples = _by("clean", side="obverse")
    assert len(samples) >= 5

    misses = []
    for path_key in samples:
        result = extract_serial_number(_read(path_key))
        got = result.get("value")
        if not got or not SERIAL_FORMAT_RE.match(got):
            misses.append((path_key, got))

    allowed = max(1, len(samples) // 5)
    assert len(misses) <= allowed, (
        f"{len(misses)} clean obverse serials failed format check "
        f"(allowed up to {allowed}): {misses}"
    )


# --------------------------------------------------------------------- fake tests


def test_structural_fakes_no_spurious_serials():
    """Structural fakes (blank, noise, half-cropped) carry no text by
    construction. OCR must never fabricate a serial on them.

    Content-degraded fakes (inverted/desaturated/hue-shifted/side-by-side)
    are deliberately excluded: they contain real banknote pixels and the
    OCR can legitimately read text from them. Distinguishing those from
    real notes is the verdict combiner's job (colour palette + structural
    sanity), not the OCR's."""
    samples = _by("fake", fake_type="structural")
    assert len(samples) >= 3

    spurious = []
    for path_key in samples:
        result = extract_serial_number(_read(path_key))
        got = result.get("value")
        if got is not None:
            spurious.append((path_key, got))

    assert spurious == [], (
        f"OCR fabricated serials on structural fakes: {spurious}"
    )
