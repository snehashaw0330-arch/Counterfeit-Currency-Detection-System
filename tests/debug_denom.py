"""Show what each corner of Rs 200 reverse contains
to find where the spurious '500' comes from."""

import os, sys
import cv2
import pytesseract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic

img = cv2.imread(os.path.join(
    ROOT, "tests", "sample_notes", "real", "real_200_rev.jpg"
))

h, w = img.shape[:2]

corners = {
    "top-left":     img[0:int(h * 0.30),         0:int(w * 0.30)],
    "top-right":    img[0:int(h * 0.30),         int(w * 0.65):w],
    "bottom-left":  img[int(h * 0.65):h,         0:int(w * 0.30)],
    "bottom-right": img[int(h * 0.65):h,         int(w * 0.55):w],
}

OUT = os.path.join(ROOT, "tests", "ocr_debug", "real_200_rev")
os.makedirs(OUT, exist_ok=True)

for label, crop in corners.items():

    cv2.imwrite(os.path.join(OUT, f"corner_{label}.jpg"), crop)
    print(f"\n========== {label}  size {crop.shape[1]}x{crop.shape[0]}")

    ch = crop.shape[0]
    min_h = ch * 2.5 * 0.20
    print(f"  min digit height = {min_h:.1f}")

    for i, variant in enumerate(forensic._preprocess_variants(crop)):
        for psm in (7, 11):
            data = pytesseract.image_to_data(
                variant,
                config=(
                    f"--oem 3 --psm {psm} "
                    "-c tessedit_char_whitelist=0123456789"
                ),
                output_type=pytesseract.Output.DICT,
            )
            for j in range(len(data["text"])):
                text = data["text"][j].strip()
                if not text:
                    continue
                try:
                    conf = int(float(data["conf"][j]))
                except (TypeError, ValueError):
                    continue
                if conf < 40:
                    continue
                height = int(data["height"][j])
                mark = " <- DENOM" if text in forensic._KNOWN_DENOMINATIONS and height >= min_h else ""
                print(f"  v{i} psm{psm}  {text!r:<14}  conf {conf:>3}  h {height:>4}{mark}")
