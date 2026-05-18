"""Measure how analyze_proportions rates real and stretched fixtures."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from backend.forensic import (
    _detect_note_quad,
    _image_borders_uniform,
    analyze_proportions,
    warmup_ocr,
)

warmup_ocr()

TARGETS = [
    ("tests/sample_notes/real_phone/real_50_new_series_specimen_0aa_obv.jpg", "50"),
    ("tests/sample_notes/real_phone/real_50_old_series_2da_obv.jpg", "50"),
    ("tests/sample_notes/real_phone/real_50_phone_obv.jpg", "50"),
    ("tests/sample_notes/real_phone/real_500_phone_obv.jpg", "500"),
    ("tests/sample_notes/real_phone/real_500_phone_handheld_obv.jpg", "500"),
    ("tests/sample_notes/real/real_50_obv.jpg", "50"),
    ("tests/sample_notes/real/real_100_obv.png", "100"),
    ("tests/sample_notes/fake/fake_11_stretched_500.jpg", "500"),
]

for path, denom in TARGETS:
    img = cv2.imread(path)
    if img is None:
        print(f"-- {path}: NO IMAGE")
        continue
    h, w = img.shape[:2]
    det = _detect_note_quad(img)
    uniform = _image_borders_uniform(img)
    result = analyze_proportions(img, denom)
    print(f"{path}")
    print(f"  frame: {w}x{h}  aspect {max(w,h)/min(w,h):.3f}")
    print(f"  quad: {'yes' if det else 'NO '}  aspect={det['aspect']:.3f}" if det else f"  quad: NO")
    print(f"  borders uniform: {uniform}")
    print(f"  status: {result['status']}  value: {result.get('value')}")
    print()
