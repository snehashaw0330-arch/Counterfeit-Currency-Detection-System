"""Debug what EasyOCR returns on the Rs 50 specimen with 0AA 000000 serial."""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from backend.forensic import (
    _ALNUM_SPACE,
    _FULL_SERIAL_NO_SPACE,
    _easyocr_words,
    _get_easyocr_reader,
    _locate_note,
    _normalize_digits,
    _normalize_prefix,
    _serial_from_words,
    extract_serial_number,
    warmup_ocr,
)

warmup_ocr()
reader = _get_easyocr_reader()

SRC = r"C:\Users\swadh\Downloads\8.jpg.jpeg"
img = cv2.imread(SRC)
print(f"raw shape: {img.shape}")

cropped = _locate_note(img)
print(f"cropped shape: {cropped.shape}")
print()

print("--- EasyOCR on raw image ---")
for bbox, text, conf in reader.readtext(img, allowlist=_ALNUM_SPACE, detail=1):
    print(f"  conf={conf:.2f}  text={text!r}")

print("\n--- EasyOCR on _locate_note crop ---")
for bbox, text, conf in reader.readtext(cropped, allowlist=_ALNUM_SPACE, detail=1):
    print(f"  conf={conf:.2f}  text={text!r}")

print("\n--- _easyocr_words (post-filter) on raw ---")
for w in _easyocr_words(img) or []:
    print(f"  {w}")

print("\n--- _easyocr_words (post-filter) on cropped ---")
for w in _easyocr_words(cropped) or []:
    print(f"  {w}")

print("\n--- Regex test on candidate tokens ---")
candidates = ["0AA000000", "OAA000000", "0AAOOOOOO", "OAAOOOOOO", "0AA 000000"]
for c in candidates:
    compact = c.replace(" ", "")
    m = _FULL_SERIAL_NO_SPACE.match(compact)
    print(f"  {c!r:18} compact={compact!r:14} match={'yes' if m else 'NO '}", end="")
    if m:
        prefix = _normalize_prefix(m.group(1))
        digits = _normalize_digits(m.group(3))
        print(f"  prefix={prefix!r}  digits={digits!r}")
    else:
        print()

print("\n--- extract_serial_number result ---")
print(extract_serial_number(img))
