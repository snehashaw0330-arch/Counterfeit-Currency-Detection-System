"""
Run pytesseract.image_to_data on the Rs 2000 bottom crop
with various PSM modes to see which one isolates the serial.
"""

import os, sys
import cv2
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic  # ensures TESSERACT_AVAILABLE side-effects

img = cv2.imread(
    os.path.join(ROOT, "tests", "ocr_debug", "real_2000_obv", "bottom.jpg")
)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

for psm in (3, 6, 7, 8, 11, 12, 13):

    print("\n=== PSM", psm, "=" * 60)

    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )

    try:
        data = pytesseract.image_to_data(
            gray, config=config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        print("  ERROR", e)
        continue

    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        try:
            c = int(float(conf))
        except (TypeError, ValueError):
            c = -1
        if c >= 30:
            print(f"  conf {c:>3}  {text!r}")
