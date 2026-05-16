"""Check what EasyOCR returns on the user's small Rs 500 image."""

import easyocr
import cv2

print("Loading EasyOCR reader (downloads detector + recogniser model "
      "on first run, ~64 MB)...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("Reader ready.\n")

img = cv2.imread(r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg")
print("Image:", img.shape)

# allowlist constrains output to alphanumerics + ₹
# (without ₹ it gets the right digits but reads ₹ as junk)
results = reader.readtext(
    img,
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    detail=1,
)

for bbox, text, conf in results:
    (x0, y0), _, (x1, y1), _ = bbox
    print(f"  conf {conf:.2f}  text {text!r:<20}  bbox ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
