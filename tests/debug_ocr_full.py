"""
Full OCR debug — try every preprocessing variant × PSM combo
on Rs 2000 bottom crop. Find what reads "0AA 000000".
"""

import os, sys
import cv2
import numpy as np
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic  # configures Tesseract path

img = cv2.imread(
    os.path.join(ROOT, "tests", "ocr_debug", "real_2000_obv", "bottom.jpg")
)


def variants(bgr):
    """Yield (label, single-channel image) preprocessing variants."""

    h, w = bgr.shape[:2]
    b, g, r = cv2.split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    for label, ch in [
        ("gray", gray),
        ("red", r),
        ("green", g),
        ("blue", b),
    ]:
        upscaled = cv2.resize(ch, None, fx=2.5, fy=2.5,
                              interpolation=cv2.INTER_CUBIC)

        yield f"{label}-raw", upscaled

        # Otsu binary
        _, otsu = cv2.threshold(upscaled, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield f"{label}-otsu", otsu

        # Inverted otsu
        _, otsu_inv = cv2.threshold(upscaled, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        yield f"{label}-otsuI", otsu_inv

        # Adaptive
        adapt = cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        yield f"{label}-adapt", adapt


hits = []

for label, im in variants(img):

    for psm in (6, 7, 11):

        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        )

        try:
            data = pytesseract.image_to_data(
                im, config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            continue

        words = []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            if not text:
                continue
            try:
                c = int(float(conf))
            except (TypeError, ValueError):
                continue
            if c >= 40:
                words.append(f"{text}({c})")

        if not words:
            continue

        joined = " ".join(words)

        # Look for the expected pattern
        has_letters = "AA" in joined or "0AA" in joined
        has_six = "000000" in joined

        marker = ""
        if has_letters and has_six:
            marker = "  ** FULL MATCH **"
        elif has_letters:
            marker = "  * has letters"
        elif has_six:
            marker = "  . has digits"

        if marker:
            hits.append((label, psm, joined, marker))


print("\n=== VARIANTS THAT FOUND SERIAL CHARACTERS ===\n")
for label, psm, words, marker in hits:
    print(f"  {label:<12} psm{psm}  {marker}")
    print(f"      {words[:200]}")
