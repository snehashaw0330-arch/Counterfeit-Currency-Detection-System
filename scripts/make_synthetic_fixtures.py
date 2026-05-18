"""Generate reproducible synthetic derivatives of real fixtures.

Run from repo root:
    venv\\Scripts\\python.exe scripts\\make_synthetic_fixtures.py

Outputs:
  - tests/sample_notes/real_phone/real_100_phone_blurry_obv.jpg
      (motion-blurred Rs 100 phone — phone-shake regression sample)
  - tests/sample_notes/fake/fake_11_stretched_50.jpg
      (Rs 50 phone stretched horizontally 30% — proportion-check
       regression sample for Phase C-1)

Deterministic, idempotent — safe to re-run.
"""

import pathlib

import cv2
import numpy as np


REAL_PHONE = pathlib.Path("tests/sample_notes/real_phone")
FAKE = pathlib.Path("tests/sample_notes/fake")


def motion_blur(img: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def horizontal_stretch(img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(
        img, (int(w * factor), h),
        interpolation=cv2.INTER_CUBIC,
    )


def main() -> None:
    blur_src = REAL_PHONE / "real_100_phone_obv.jpg"
    blur_dst = REAL_PHONE / "real_100_phone_blurry_obv.jpg"
    img = cv2.imread(str(blur_src))
    if img is None:
        raise SystemExit(f"could not read {blur_src}")
    cv2.imwrite(
        str(blur_dst), motion_blur(img, kernel_size=9),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    print(f"wrote {blur_dst} ({blur_dst.stat().st_size} bytes)")

    # Stretch source: the clean Wikipedia Rs 100 obverse,
    # because _detect_note_quad reliably finds its quad
    # (verified empirically — phone fixtures with gradient
    # backgrounds or small frames don't give consistent
    # contour detection). A 30% horizontal stretch takes the
    # quad aspect from 2.343 to ~3.05, well above the 15%
    # tolerance, so the proportion check fires.
    stretch_src = pathlib.Path("tests/sample_notes/real/real_100_obv.png")
    stretch_dst = FAKE / "fake_11_stretched_100.jpg"
    img = cv2.imread(str(stretch_src))
    if img is None:
        raise SystemExit(f"could not read {stretch_src}")
    cv2.imwrite(
        str(stretch_dst), horizontal_stretch(img, factor=1.3),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    print(f"wrote {stretch_dst} ({stretch_dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
