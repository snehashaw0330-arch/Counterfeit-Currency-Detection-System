"""Probe what Tesseract reads on the whole user-supplied Rs 500 image
at multiple scales — to design a layout-independent OCR pipeline."""

import os, sys
import cv2
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend import forensic  # configures Tesseract path

img = cv2.imread(r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg")
print("Original shape:", img.shape)

# Resize to several target widths so we can see where Tesseract
# starts and stops reading reliably.
for target_w in (800, 1200, 1600, 2000):
    scale = target_w / img.shape[1]
    resized = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    print(f"\n==== resized to {resized.shape[1]}x{resized.shape[0]} ====")

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = forensic._CLAHE.apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for psm in (6, 11):
        data = pytesseract.image_to_data(
            otsu,
            config=f"--oem 3 --psm {psm} "
                   "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
            output_type=pytesseract.Output.DICT,
        )
        words = []
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            if not t:
                continue
            try:
                c = int(float(data["conf"][i]))
            except (TypeError, ValueError):
                continue
            if c < 40:
                continue
            words.append(f"{t}({c})")
        if words:
            print(f"  psm{psm}: {' '.join(words[:30])[:300]}")
