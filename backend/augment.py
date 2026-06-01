"""
Seeded image augmentation for Phase D training.

Our labelled base corpus is small (~34 images), so we expand it
by generating photometric + geometric variants of each TRAINING
image. This is standard data augmentation — it does not change an
image's class (a jittered genuine note is still genuine) — and it
is the honest, defensible form of "generative" data expansion for
a detection model (we never synthesise realistic counterfeit
currency; see docs/PROJECT_SCOPE.md).

Contract:
  - ``augment_variants(bgr, n, seed) -> list[np.ndarray]`` returns
    exactly ``n`` augmented BGR images (the original is NOT
    included; the caller adds it separately).
  - Deterministic: same (image, n, seed) => byte-identical output.
    Every model run is therefore reproducible.
  - Never raises: unusable input yields ``[]``.
"""

import numpy as np
import cv2

from backend.forensic import _ensure_bgr


def _rotate(img, rng):
    h, w = img.shape[:2]
    angle = float(rng.uniform(-10, 10))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), borderMode=cv2.BORDER_REPLICATE
    )


def _brightness_contrast(img, rng):
    alpha = float(rng.uniform(0.8, 1.2))   # contrast gain
    beta = float(rng.uniform(-25, 25))     # brightness offset
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _blur(img, rng):
    k = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(img, (k, k), 0)


def _noise(img, rng):
    sigma = float(rng.uniform(5, 18))
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg(img, rng):
    q = int(rng.integers(40, 85))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def _scale_crop(img, rng):
    h, w = img.shape[:2]
    s = float(rng.uniform(1.0, 1.15))
    nh, nw = max(int(h * s), h), max(int(w * s), w)
    big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y = int(rng.integers(0, max(nh - h, 1)))
    x = int(rng.integers(0, max(nw - w, 1)))
    return big[y:y + h, x:x + w]


def _perspective(img, rng):
    h, w = img.shape[:2]
    m = 0.04

    def jit(px, py):
        return [px + float(rng.uniform(-m, m)) * w,
                py + float(rng.uniform(-m, m)) * h]

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([jit(0, 0), jit(w, 0), jit(w, h), jit(0, h)])
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, transform, (w, h), borderMode=cv2.BORDER_REPLICATE
    )


_TRANSFORMS = [
    _rotate, _brightness_contrast, _blur,
    _noise, _jpeg, _scale_crop, _perspective,
]


def augment_variants(image, n, seed=0):
    """Return ``n`` seeded augmented variants of ``image`` (BGR).

    Each variant applies a random subset of 2–4 transforms in a
    random order. Deterministic for a given (image, n, seed).
    Returns [] on unusable input."""

    try:
        bgr = _ensure_bgr(image)
    except Exception:
        return []

    if n <= 0:
        return []

    out = []
    for i in range(n):
        # Distinct, reproducible stream per variant.
        rng = np.random.default_rng(seed * 100003 + i)
        img = bgr.copy()
        k = int(rng.integers(2, 5))
        chosen = rng.choice(len(_TRANSFORMS), size=k, replace=False)
        for ti in chosen:
            try:
                img = _TRANSFORMS[int(ti)](img, rng)
            except Exception:
                pass  # a failed transform just leaves the image as-is
        out.append(img)

    return out
