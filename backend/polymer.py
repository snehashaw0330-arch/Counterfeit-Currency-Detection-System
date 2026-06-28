"""
Polymer-banknote heuristics.

These checks are intentionally conservative:
- PASS only when there is strong positive evidence of polymer traits.
- INFO otherwise.
- Never FAIL on substrate alone; absence of a cue in a single photograph is not
  evidence of a paper counterfeit.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_bgr(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image shape: {arr.shape}")


def _window_candidates(hsv: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    h, w = sat.shape[:2]
    if h < 20 or w < 20:
        return []

    bright_low_sat = ((sat < 42) & (val > 172)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(bright_low_sat, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = []
    img_area = float(h * w)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        area_ratio = area / img_area
        if area_ratio < 0.004 or area_ratio > 0.18:
            continue
        if x <= 1 or y <= 1 or x + ww >= w - 1 or y + hh >= h - 1:
            continue

        roi_sat = sat[y:y + hh, x:x + ww]
        roi_val = val[y:y + hh, x:x + ww]
        mean_sat = float(roi_sat.mean())
        mean_val = float(roi_val.mean())

        pad = max(3, int(min(ww, hh) * 0.1))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)
        ring = val[y0:y1, x0:x1].copy()
        ring[pad:pad + hh, pad:pad + ww] = 0
        ring_nonzero = ring[ring > 0]
        ring_val = float(ring_nonzero.mean()) if ring_nonzero.size else mean_val

        edge_contrast = max(0.0, mean_val - ring_val)
        slender = max(ww, hh) / max(min(ww, hh), 1)
        shape_bonus = 0.0
        if slender >= 1.6:
            shape_bonus += 0.35
        if hh >= int(h * 0.22):
            shape_bonus += 0.2

        score = (
            max(0.0, (190.0 - mean_sat) / 190.0) * 0.45
            + max(0.0, (mean_val - 160.0) / 95.0) * 0.3
            + min(0.35, edge_contrast / 70.0) * 0.25
            + shape_bonus
        )
        out.append((float(score), (int(x), int(y), int(ww), int(hh))))

    out.sort(key=lambda item: item[0], reverse=True)
    return out


def detect_transparent_window(image) -> dict:
    try:
        bgr = _ensure_bgr(image)
    except Exception:
        return {"status": "INFO", "details": "Image format unsupported", "value": None}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cands = _window_candidates(hsv)
    if not cands:
        return {
            "status": "INFO",
            "details": "No strong transparent-window candidate found in this view",
            "value": None,
        }

    score, bbox = cands[0]
    if score < 0.95:
        return {
            "status": "INFO",
            "details": "A bright low-saturation patch exists, but not strongly enough to confirm a polymer window",
            "value": {"window_score": round(score, 4), "bbox": bbox},
        }

    return {
        "status": "PASS",
        "details": "Detected a strong transparent-window candidate consistent with a polymer note",
        "value": {"window_score": round(score, 4), "bbox": bbox},
    }


def detect_substrate_sheen(image) -> dict:
    try:
        bgr = _ensure_bgr(image)
    except Exception:
        return {"status": "INFO", "details": "Image format unsupported", "value": None}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    highlight = ((val > 235) & (sat < 70)).astype(np.uint8)
    if highlight.sum() == 0:
        return {
            "status": "INFO",
            "details": "No reliable specular highlight was visible; sheen cannot be confirmed from this image",
            "value": None,
        }

    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    residual = cv2.subtract(gray, blur)
    smoothness = float(np.mean(np.abs(residual[highlight > 0]))) if np.any(highlight) else 0.0
    coverage = float(highlight.mean())
    glossy_score = coverage * 12.0 + max(0.0, 9.0 - smoothness) / 9.0

    if glossy_score < 0.75:
        return {
            "status": "INFO",
            "details": "Some bright regions exist, but the substrate sheen is not strong enough to confirm polymer from a single view",
            "value": {
                "sheen_score": round(glossy_score, 4),
                "highlight_coverage": round(coverage, 4),
            },
        }

    return {
        "status": "PASS",
        "details": "Visible localized highlights are consistent with glossy polymer substrate",
        "value": {
            "sheen_score": round(glossy_score, 4),
            "highlight_coverage": round(coverage, 4),
        },
    }


def analyze_polymer_features(image) -> dict:
    window = detect_transparent_window(image)
    sheen = detect_substrate_sheen(image)
    overall = "PASS" if "PASS" in {window["status"], sheen["status"]} else "INFO"
    return {
        "status": overall,
        "details": (
            "Strong polymer cue detected" if overall == "PASS"
            else "No strong polymer cue confirmed from this single image"
        ),
        "value": {"transparent_window": window, "substrate_sheen": sheen},
    }
