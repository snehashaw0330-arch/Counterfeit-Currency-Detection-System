"""
Save the OCR top-left + bottom-right crops for every real
note to tests/ocr_debug/<file>/{top.jpg,bottom.jpg}.

Used to visually inspect what Tesseract is actually seeing
on each banknote before tuning preprocessing.
"""

import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC = os.path.join(ROOT, "tests", "sample_notes", "real")
OUT = os.path.join(ROOT, "tests", "ocr_debug")
os.makedirs(OUT, exist_ok=True)

for name in sorted(os.listdir(SRC)):

    if not name.startswith("real_"):
        continue

    path = os.path.join(SRC, name)
    img = cv2.imread(path)
    if img is None:
        # PNG with alpha → load via cv2 with UNCHANGED then drop alpha
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img is None:
        print(f"SKIP {name} (cannot read)")
        continue

    h, w = img.shape[:2]

    top = img[int(h * 0.12):int(h * 0.32), int(w * 0.03):int(w * 0.32)]
    bot = img[int(h * 0.78):h,             int(w * 0.55):w]

    sub = os.path.join(OUT, os.path.splitext(name)[0])
    os.makedirs(sub, exist_ok=True)

    cv2.imwrite(os.path.join(sub, "top.jpg"), top)
    cv2.imwrite(os.path.join(sub, "bottom.jpg"), bot)
    print(f"{name:<32}  full {w}x{h}  top {top.shape[1]}x{top.shape[0]}  "
          f"bot {bot.shape[1]}x{bot.shape[0]}")

print(f"\nCrops written to {OUT}")
