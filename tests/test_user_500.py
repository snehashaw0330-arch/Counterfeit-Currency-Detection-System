"""Run the full forensic pipeline on the user-supplied small Rs 500 image."""

import os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic

img = cv2.imread(r"C:\Users\swadh\Downloads\images.note 500.jpg.jpeg")
print(f"Input: {img.shape[1]}x{img.shape[0]}")

print("\n--- OCR serial ---")
print(forensic.extract_serial_number(img))

print("\n--- Denomination ---")
print(forensic.classify_denomination(img))
