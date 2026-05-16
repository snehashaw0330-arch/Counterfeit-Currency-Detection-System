"""See what denomination tokens are detected on the user Rs 500 image."""

import os, sys
import cv2
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend import forensic

img = cv2.imread(r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg")
img = forensic._normalise_for_ocr(forensic._ensure_bgr(img))
print("Normalised:", img.shape[1], "x", img.shape[0])

h, w = img.shape[:2]
corners = [
    ("whole",        img),
    ("top-left",     img[0:int(h*0.30), 0:int(w*0.30)]),
    ("top-right",    img[0:int(h*0.30), int(w*0.65):w]),
    ("bottom-left",  img[int(h*0.65):h, 0:int(w*0.30)]),
    ("bottom-right", img[int(h*0.65):h, int(w*0.55):w]),
]

for label, crop in corners:
    print(f"\n========== {label}  {crop.shape[1]}x{crop.shape[0]}")
    min_frac = 0.05 if label == "whole" else 0.20
    for vi, variant in enumerate(forensic._preprocess_variants(crop)):
        ph = variant.shape[0]
        min_h = ph * min_frac
        for psm in (7, 11):
            data = pytesseract.image_to_data(
                variant,
                config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT,
            )
            for i in range(len(data["text"])):
                t = data["text"][i].strip()
                if not t:
                    continue
                try:
                    c = int(float(data["conf"][i]))
                except (TypeError, ValueError):
                    continue
                if c < 30:
                    continue
                hh = int(data["height"][i])
                ok = (t in forensic._KNOWN_DENOMINATIONS) and hh >= min_h
                mark = "  <-- DENOM HIT" if ok else ""
                print(f"  v{vi} psm{psm}  {t!r:<10}  c{c:>3}  h{hh:>4}  min_h{min_h:.0f}{mark}")
