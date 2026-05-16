"""Trace exactly what _ocr_words returns at each step on the user image."""

import os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic

img = cv2.imread(r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg")
print(f"Input: {img.shape[1]}x{img.shape[0]}")

img = forensic._normalise_for_ocr(forensic._ensure_bgr(img))
print(f"After normalise: {img.shape[1]}x{img.shape[0]}")

h = img.shape[0]
regions = [
    ("whole",  img),
    ("top",    img[0:int(h * 0.45), :]),
    ("bottom", img[int(h * 0.55):h, :]),
]

for label, region in regions:
    print(f"\n========== {label}  {region.shape[1]}x{region.shape[0]}")
    for vi, variant in enumerate(forensic._preprocess_variants(region)):
        print(f"  variant {vi}  -> {variant.shape[1]}x{variant.shape[0]}")
        for psm in (6, 11):
            words = forensic._ocr_words(variant, psm)
            keep = [w for w in words if w["conf"] >= 50 and len(w["text"]) >= 2]
            if keep:
                tokens = " | ".join(f"{w['text']}({w['conf']})" for w in keep[:12])
                print(f"    psm{psm}: {tokens[:250]}")
