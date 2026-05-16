"""
Iterate over different crop coordinate guesses and run
Tesseract on each, to find the tightest crop that reliably
isolates the serial number band on Indian banknotes.

For each candidate (top_y0, top_y1, bottom_y0, bottom_y1)
we report how many of the 10 real notes yield a detected
serial in the expected format.
"""

import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic  # noqa: E402

SRC = os.path.join(ROOT, "tests", "sample_notes", "real")

# Accept any digit+letter+letter+digits style with mild OCR fudge.
# We're tuning *which crop region* works best — not regex strictness.
PATTERN = re.compile(r"([0-9OISB])([A-Z08]{2})\s*([0-9OISB]{6,7})")


def load(path):
    img = cv2.imread(path)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def try_ocr(crop):

    if crop.size == 0:
        return []

    found = []

    for variant in forensic._preprocess_variants(crop):

        for psm in (7, 6, 8):

            text = forensic._ocr_region(variant, psm=psm)

            for m in PATTERN.finditer(text):
                found.append(
                    f"{m.group(1)}{m.group(2)} {m.group(3)}"
                )

    return found


# Crop candidates  (top_y0, top_y1, top_x0, top_x1,
#                   bot_y0, bot_y1, bot_x0, bot_x1)
# Each as fractions of the full image height/width.
CANDIDATES = [
    # Current (loose) crops
    (0.12, 0.32, 0.03, 0.32, 0.78, 1.00, 0.55, 1.00),
    # Tighter — only the serial band
    (0.15, 0.27, 0.03, 0.28, 0.83, 0.96, 0.60, 0.99),
    # Tighter still — single line strip
    (0.17, 0.25, 0.03, 0.28, 0.86, 0.95, 0.60, 0.98),
    # Slightly wider tail
    (0.15, 0.30, 0.02, 0.32, 0.82, 0.97, 0.55, 1.00),
]


def evaluate(candidate):

    ty0, ty1, tx0, tx1, by0, by1, bx0, bx1 = candidate
    hits = 0
    total = 0

    for name in sorted(os.listdir(SRC)):

        if not name.endswith((".jpg", ".png")):
            continue

        if "_rev" in name:
            # Reverse sides genuinely have no serial — skip.
            continue

        path = os.path.join(SRC, name)
        img = load(path)
        if img is None:
            continue

        h, w = img.shape[:2]

        top = img[int(h * ty0):int(h * ty1), int(w * tx0):int(w * tx1)]
        bot = img[int(h * by0):int(h * by1), int(w * bx0):int(w * bx1)]

        cands = try_ocr(top) + try_ocr(bot)

        total += 1
        if cands:
            hits += 1
            print(f"  {name:<24} -> {cands[:3]}")
        else:
            print(f"  {name:<24} -> MISS")

    return hits, total


for cand in CANDIDATES:
    print("\n" + "=" * 70)
    print(f"Candidate top y={cand[0]:.2f}-{cand[1]:.2f} "
          f"x={cand[2]:.2f}-{cand[3]:.2f}  /  "
          f"bot y={cand[4]:.2f}-{cand[5]:.2f} "
          f"x={cand[6]:.2f}-{cand[7]:.2f}")
    print("=" * 70)
    hits, total = evaluate(cand)
    print(f"  >>> HITS {hits}/{total}")
