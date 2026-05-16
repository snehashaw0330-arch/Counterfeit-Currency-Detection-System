"""
Generate 10 fake / negative-control samples by perturbing
a real reference note. Each variant simulates a failure mode
that a counterfeit detector must not approve:

  blank_black, blank_white, pure_noise: no-content controls
  heavy_blur:          a photocopied/scanned-of-a-scan fake
  low_res:             downsampled then upsampled (print quality loss)
  half_cropped:        only top half of a note (incomplete)
  color_inverted:      colours flipped (printer ink mismatch)
  desaturated:         grayscale photocopy
  hue_shifted:         strong colour shift (wrong ink)
  side_by_side_user:   copy of the user's annotated comparison image
"""

import os

import cv2
import numpy as np

SCRIPT = os.path.dirname(os.path.abspath(__file__))

REAL_DIR = os.path.join(SCRIPT, "sample_notes", "real")
FAKE_DIR = os.path.join(SCRIPT, "sample_notes", "fake")
os.makedirs(FAKE_DIR, exist_ok=True)

# Pick a real note as the base for perturbation fakes.
base = cv2.imread(os.path.join(REAL_DIR, "real_500.jpg"))

if base is None:
    # Fall back to the 2000 obverse if 500 isn't there yet.
    base = cv2.imread(os.path.join(REAL_DIR, "real_2000_obv.jpg"))

if base is None:
    raise SystemExit("No reference real note found in tests/sample_notes/real/")

H, W = base.shape[:2]

# 1. blank black
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_01_blank_black.jpg"),
    np.zeros((H, W, 3), dtype=np.uint8),
)

# 2. blank white
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_02_blank_white.jpg"),
    np.full((H, W, 3), 255, dtype=np.uint8),
)

# 3. pure noise
rng = np.random.default_rng(0)
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_03_pure_noise.jpg"),
    rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
)

# 4. heavy blur (kernel 51) — photocopy of a photocopy
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_04_heavy_blur.jpg"),
    cv2.GaussianBlur(base, (51, 51), 0),
)

# 5. low-res destruction (downsample to 80px width, back up)
small = cv2.resize(base, (80, int(80 * H / W)))
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_05_low_res.jpg"),
    cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST),
)

# 6. half-cropped (only top half)
half = base.copy()
half[H // 2:] = 0
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_06_half_cropped.jpg"),
    half,
)

# 7. colour-inverted
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_07_color_inverted.jpg"),
    cv2.bitwise_not(base),
)

# 8. desaturated (grey photocopy)
gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_08_desaturated.jpg"),
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
)

# 9. strong hue shift (wrong ink colour)
hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.int32)
hsv[:, :, 0] = (hsv[:, :, 0] + 60) % 180
cv2.imwrite(
    os.path.join(FAKE_DIR, "fake_09_hue_shifted.jpg"),
    cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR),
)

# 10. user-supplied annotated side-by-side comparison image
src_user = r"C:\Users\swadh\Downloads\download.jpg"
if os.path.exists(src_user):
    img = cv2.imread(src_user)
    if img is not None:
        cv2.imwrite(
            os.path.join(FAKE_DIR, "fake_10_user_sidebyside.jpg"),
            img,
        )

print("Wrote", len(os.listdir(FAKE_DIR)), "fake samples to", FAKE_DIR)
