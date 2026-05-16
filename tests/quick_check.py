"""Quick check on a handful of Wikipedia samples + user image."""

import os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend import forensic

cases = [
    r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg",
    os.path.join(ROOT, "tests/sample_notes/real/real_50_obv.jpg"),
    os.path.join(ROOT, "tests/sample_notes/real/real_200_obv.jpg"),
    os.path.join(ROOT, "tests/sample_notes/real/real_2000_obv.jpg"),
    os.path.join(ROOT, "tests/sample_notes/real/real_10_obv.jpg"),
    os.path.join(ROOT, "tests/sample_notes/real/real_20_obv.jpg"),
]

for path in cases:
    name = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        print(f"  {name:<32} (cannot read)")
        continue
    ocr = forensic.extract_serial_number(img)
    den = forensic.classify_denomination(img)
    print(f"  {name:<32}  OCR={ocr.get('value')!r:<28}  DEN={den.get('value')!r}")
