"""Debug what EasyOCR returns on the Rs 50 with asterisk-separated serial."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from backend.forensic import (
    _ALNUM_SPACE,
    _get_easyocr_reader,
    _locate_note,
    extract_serial_number,
    warmup_ocr,
)

warmup_ocr()
reader = _get_easyocr_reader()

SRC = r"C:\Users\swadh\Downloads\9.jpg.jpeg"
img = cv2.imread(SRC)
print(f"raw shape: {img.shape}")

cropped = _locate_note(img)
print(f"cropped shape: {cropped.shape}")

print("\n--- EasyOCR with current allowlist (alnum + space, no asterisk) ---")
for bbox, text, conf in reader.readtext(img, allowlist=_ALNUM_SPACE, detail=1):
    print(f"  conf={conf:.2f}  text={text!r}")

print("\n--- EasyOCR with asterisk added to allowlist ---")
allowlist_with_star = _ALNUM_SPACE + "*"
for bbox, text, conf in reader.readtext(img, allowlist=allowlist_with_star, detail=1):
    print(f"  conf={conf:.2f}  text={text!r}")

print("\n--- EasyOCR with NO allowlist (sees everything) ---")
for bbox, text, conf in reader.readtext(img, detail=1):
    print(f"  conf={conf:.2f}  text={text!r}")

print("\n--- current extract_serial_number result ---")
print(extract_serial_number(img))
