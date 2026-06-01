"""
Hand-crafted visual feature extraction for the classical machine
learning techniques in Phase D (see docs/PROJECT_SCOPE.md §4).

This module turns a banknote image into a FIXED-LENGTH numeric
feature vector built only from fast, deterministic, visible-light
image statistics — no OCR, no neural network. It mirrors the
approach in the reference thesis (Local Binary Pattern histograms
over the note) and adds colour-distribution and structural
descriptors so the classical models (SVM / RandomForest / KNN /
LogisticRegression) have a meaningful, reproducible input.

Design contract:
  - ``extract_feature_vector(bgr) -> np.ndarray`` of shape
    ``(FEATURE_DIM,)``, dtype float32.
  - Deterministic: the same image in produces the same vector out.
  - Never raises: unusable input yields a best-effort / zero-filled
    vector of the correct length, so one bad sample can't abort a
    training run over thousands of images.
  - No train/serve skew: the SAME function builds the training
    matrix and featurises an image at inference time.

The vector is intentionally independent of the forensic rule
checks and the CNN so the three views — CNN on raw pixels,
classical ML on these hand-crafted features, forensic rules on
security features — stay statistically independent for the
Phase D fusion layer.

Feature values are returned RAW (unscaled). Standardisation
(zero-mean / unit-variance) is the training pipeline's job
(``StandardScaler``), kept there so the scaler is fit on the
training split only and persisted alongside the model.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern

from backend.forensic import (
    _ensure_bgr, _locate_note, _microprint_score, _watermark_variance,
)

# ---- Local Binary Pattern: matches the reference thesis
# (radius 3, 8*radius sample points, uniform method). Uniform
# LBP with P points yields P+2 bins.
_LBP_RADIUS = 3
_LBP_POINTS = 8 * _LBP_RADIUS          # 24
_LBP_BINS = _LBP_POINTS + 2            # 26

_HUE_BINS = 12                         # coarse hue distribution
_CANON_W = 384                         # canonical landscape size for
_CANON_H = 192                         # texture/colour comparability
_SAT_MASK = 40                         # "coloured pixel" saturation floor

# Fixed feature layout. The order here is the contract — training
# and inference both rely on it, and feature-importance reports
# map back through these names.
FEATURE_NAMES = (
    [f"lbp_{i}" for i in range(_LBP_BINS)]
    + [f"hue_{i}" for i in range(_HUE_BINS)]
    + ["mean_sat", "std_sat", "mean_val", "std_val", "hue_entropy"]
    + ["aspect", "edge_density", "gray_mean", "gray_std"]
    # OCR-free forensic security signals folded into the ML vector
    # (Phase E -> ML): fine-print sharpness, watermark-panel gradient
    # variance, and central vertical-edge energy (security-thread proxy).
    # log1p compresses the heavy-tailed sharpness/variance ranges.
    + ["microprint_logsharp", "watermark_logvar", "vthread_energy"]
)
FEATURE_DIM = len(FEATURE_NAMES)       # 50


def _lbp_hist(gray):
    """Uniform-LBP histogram (density-normalised) — 26 bins."""

    lbp = local_binary_pattern(
        gray, _LBP_POINTS, _LBP_RADIUS, method="uniform"
    )
    hist, _ = np.histogram(
        lbp, bins=_LBP_BINS, range=(0, _LBP_BINS), density=True
    )
    return hist.astype(np.float32)


def _colour_feats(bgr):
    """Return (hue_hist[12], scalars[5]).

    scalars = [mean_sat, std_sat, mean_val, std_val, hue_entropy]."""

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    coloured = hue[sat > _SAT_MASK]

    if coloured.size >= 50:
        hue_hist, _ = np.histogram(
            coloured, bins=_HUE_BINS, range=(0, 180), density=True
        )
        # Shannon entropy of the full 180-bin hue distribution —
        # rich palettes score high, photocopies/desaturated low.
        full = np.bincount(coloured, minlength=180).astype(np.float64)
        full /= full.sum()
        nz = full[full > 0]
        hue_entropy = float(-(nz * np.log2(nz)).sum())
    else:
        hue_hist = np.zeros(_HUE_BINS, dtype=np.float64)
        hue_entropy = 0.0

    scalars = np.array([
        float(sat.mean()), float(sat.std()),
        float(val.mean()), float(val.std()),
        hue_entropy,
    ], dtype=np.float32)

    return hue_hist.astype(np.float32), scalars


def _forensic_extras(note, gray):
    """OCR-free forensic security signals as 3 numeric features:
    [microprint_logsharp, watermark_logvar, vthread_energy].

    Reuses the live forensic helpers so train/serve stay aligned.
    log1p compresses the heavy-tailed sharpness/variance ranges so
    a single very-high-detail note doesn't dominate the scaler."""

    try:
        sharp, _ = _microprint_score(note)
    except Exception:
        sharp = 0.0

    try:
        wv = _watermark_variance(note)
        wv = 0.0 if wv is None else wv
    except Exception:
        wv = 0.0

    # Vertical-edge energy in the central strip — a cheap, scale-
    # stable security-thread proxy (the thread is a tall vertical
    # feature). Computed on the canonical gray so it's resolution-
    # normalised like the rest of the vector.
    try:
        h, w = gray.shape[:2]
        strip = gray[:, int(w * 0.30):int(w * 0.70)]
        sob = cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)
        vthread = float(np.mean(np.abs(sob)))
    except Exception:
        vthread = 0.0

    return np.array([
        float(np.log1p(max(sharp, 0.0))),
        float(np.log1p(max(wv, 0.0))),
        vthread,
    ], dtype=np.float32)


def extract_feature_vector(image):
    """Featurise a BGR image into a length-``FEATURE_DIM`` vector.

    Never raises. On catastrophic failure returns a zero vector of
    the correct length so a single bad sample doesn't abort a
    batch training run."""

    try:
        bgr = _ensure_bgr(image)
    except Exception:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Aspect is measured on the located note BEFORE the canonical
    # resize, so it reflects real note geometry, not the resize.
    try:
        note = _locate_note(bgr)
    except Exception:
        note = bgr

    h0, w0 = note.shape[:2]
    aspect = float(max(w0, h0) / max(min(w0, h0), 1))

    try:
        canon = cv2.resize(
            note, (_CANON_W, _CANON_H), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    except Exception:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # ---- LBP texture (26) ----
    try:
        lbp = _lbp_hist(gray)
    except Exception:
        lbp = np.zeros(_LBP_BINS, dtype=np.float32)

    # ---- Colour distribution (12 + 5) ----
    try:
        hue_hist, colour_scalars = _colour_feats(canon)
    except Exception:
        hue_hist = np.zeros(_HUE_BINS, dtype=np.float32)
        colour_scalars = np.zeros(5, dtype=np.float32)

    # ---- Structure (4) ----
    try:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        gray_mean = float(gray.mean())
        gray_std = float(gray.std())
    except Exception:
        edge_density = gray_mean = gray_std = 0.0

    struct = np.array(
        [aspect, edge_density, gray_mean, gray_std], dtype=np.float32
    )

    # ---- OCR-free forensic security signals (3) ----
    try:
        extras = _forensic_extras(note, gray)
    except Exception:
        extras = np.zeros(3, dtype=np.float32)

    vec = np.concatenate([lbp, hue_hist, colour_scalars, struct, extras])

    # Guard against any NaN/inf leaking into the training matrix.
    vec = np.nan_to_num(
        vec, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

    # Defensive length check — the contract is FEATURE_DIM.
    if vec.shape[0] != FEATURE_DIM:
        out = np.zeros(FEATURE_DIM, dtype=np.float32)
        out[: min(vec.shape[0], FEATURE_DIM)] = vec[:FEATURE_DIM]
        return out

    return vec
