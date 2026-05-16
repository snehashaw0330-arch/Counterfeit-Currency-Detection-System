"""Derive a motion-blurred copy of a fixture to simulate handheld phone shake.

Run from repo root:
    venv\Scripts\python.exe scripts\make_blurry_fixture.py

Produces tests/sample_notes/real_phone/real_100_phone_blurry_obv.jpg from
real_100_phone_obv.jpg. Deterministic; safe to re-run.
"""

import pathlib

import cv2
import numpy as np


SRC = pathlib.Path("tests/sample_notes/real_phone/real_100_phone_obv.jpg")
DST = pathlib.Path("tests/sample_notes/real_phone/real_100_phone_blurry_obv.jpg")


def motion_blur(img, kernel_size: int = 15) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def main() -> None:
    img = cv2.imread(str(SRC))
    if img is None:
        raise SystemExit(f"could not read {SRC}")

    blurred = motion_blur(img, kernel_size=9)
    cv2.imwrite(str(DST), blurred, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
