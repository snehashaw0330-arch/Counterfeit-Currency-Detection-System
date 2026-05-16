"""Dump raw EasyOCR output for fixtures whose denom/serial regressed."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from backend.forensic import _ALNUM_SPACE, _DIGITS_ALLOWLIST, _get_easyocr_reader, warmup_ocr

warmup_ocr()
reader = _get_easyocr_reader()

TARGETS = [
    "tests/sample_notes/real_phone/real_20_phone_specimen_obv.jpg",
    "tests/sample_notes/real_phone/real_50_phone_obv.jpg",
    "tests/sample_notes/real_phone/real_500_phone_handheld_obv.jpg",
]

for path in TARGETS:
    img = cv2.imread(path)
    if img is None:
        print(f"-- {path}: NO IMAGE")
        continue
    print(f"\n=== {path}  shape={img.shape} ===")
    print("[alpha+digit allowlist]")
    for bbox, text, conf in reader.readtext(img, allowlist=_ALNUM_SPACE, detail=1):
        print(f"  conf={conf:.2f}  text={text!r}")
    print("[digits-only allowlist]")
    for bbox, text, conf in reader.readtext(img, allowlist=_DIGITS_ALLOWLIST, detail=1):
        print(f"  conf={conf:.2f}  text={text!r}")
