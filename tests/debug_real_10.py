"""See what image_to_data returns for the Rs 10 obverse crops."""

import os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic

img = cv2.imread(os.path.join(
    ROOT, "tests", "sample_notes", "real", "real_10_obv.jpg"
))
h, w = img.shape[:2]

crops = {
    "top":    img[int(h * 0.12):int(h * 0.32), int(w * 0.03):int(w * 0.32)],
    "bottom": img[int(h * 0.78):h,             int(w * 0.55):w],
}

for label, crop in crops.items():
    print(f"\n========== {label} ==========")
    for i, variant in enumerate(forensic._preprocess_variants(crop)):
        for psm in (6, 7, 8, 11):
            words = forensic._ocr_words(variant, psm)
            interesting = [
                w for w in words
                if any(c.isdigit() for c in w["text"])
                or any(c.isalpha() for c in w["text"])
            ]
            if not interesting:
                continue
            tag = f"variant {i} psm{psm}"
            tokens = " | ".join(
                f"{w['text']}({w['conf']})" for w in interesting
            )
            print(f"  {tag:<18}  {tokens[:200]}")
