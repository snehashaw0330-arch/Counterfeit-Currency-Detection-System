"""
Diagnostic harness for the counterfeit detection pipeline.

Runs every image under tests/sample_notes/{real,fake}/ through
both the forensic pipeline and the full /predict logic, then
emits:

  1. A confusion-matrix-style summary on stdout (REAL/FAKE rows).
  2. A per-check pass/fail table per sample.
  3. A diagnostic dump (raw OCR text on each crop, raw numbers
     for watermark variance, security thread count, hologram
     score) in tests/diagnostic_report.json.

This is intentionally NOT a pytest file — it's an investigative
script. Run it directly with:

    venv\Scripts\python tests\diagnostic_harness.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np

# Make `backend` importable when running this file directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import forensic  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402
from PIL import Image  # noqa: E402
import io  # noqa: E402


# =====================================================
# CONFIG
# =====================================================

SAMPLES_DIR = os.path.join(ROOT, "tests", "sample_notes")
REPORT_PATH = os.path.join(ROOT, "tests", "diagnostic_report.json")

MODEL_PATH = os.path.join(
    ROOT, "models", "mobilenet_counterfeit_detector.keras"
)


# =====================================================
# LOAD MODEL ONCE
# =====================================================

print("Loading MobileNetV2 ...", flush=True)
model = load_model(MODEL_PATH)
print("Model loaded.\n", flush=True)


# =====================================================
# REPLICATE THE /predict PIPELINE LOCALLY
# =====================================================

def run_full_pipeline(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    model_input = cv2.resize(rgb, (224, 224)) / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    raw = float(model.predict(model_input, verbose=0)[0][0])

    fa = forensic.run_forensic_pipeline(bgr)

    scored = [
        c for k, c in fa.items()
        if k != "modular_ai_pipeline"
        and c["status"] in ("PASS", "FAIL")
    ]

    pass_count = sum(1 for c in scored if c["status"] == "PASS")
    total = max(len(scored), 1)
    forensic_score = pass_count / total

    combined = 0.4 * raw + 0.6 * forensic_score

    sanity = fa.get("structural_sanity", {})
    colour = fa.get("hologram_detection", {})
    proportion = fa.get("proportion_analysis", {})

    if sanity.get("status") == "FAIL":
        verdict = "FAKE"
    elif (
        combined >= 0.65
        and pass_count >= 5
        and colour.get("status") != "FAIL"
        and proportion.get("status") != "FAIL"
    ):
        verdict = "REAL"
    elif combined < 0.35 or (raw < 0.35 and forensic_score < 0.35):
        verdict = "FAKE"
    else:
        verdict = "SUSPICIOUS"

    return {
        "raw": raw,
        "forensic_score": forensic_score,
        "pass_count": pass_count,
        "total": total,
        "combined": combined,
        "verdict": verdict,
        "forensic_analysis": fa,
    }


# =====================================================
# CAPTURE RAW NUMBERS PER SAMPLE
# =====================================================
# We don't trust the PASS/FAIL labels alone — we want the
# underlying numeric signal so we can decide whether the
# thresholds are correct.

def diagnostic_signals(image_path):

    bgr = cv2.imread(image_path)
    if bgr is None:
        # PNG with alpha needs PIL fallback
        pil = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]

    # Watermark panel variance
    panel = bgr[int(h * 0.15):int(h * 0.85), 0:int(w * 0.25)]
    blurred = cv2.GaussianBlur(
        cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), (21, 21), 0
    )
    wm_variance = float(np.var(blurred))

    # Saturation ratio (UV proxy)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    high_sat = cv2.inRange(hsv, (0, 120, 120), (179, 255, 255))
    uv_ratio = float(np.count_nonzero(high_sat)) / high_sat.size

    # Security thread (vertical line count in central strip)
    strip = bgr[0:h, int(w * 0.35):int(w * 0.65)]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(h * 0.4), maxLineGap=15,
    )

    vert_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 8 and abs(y2 - y1) > h * 0.3:
                vert_count += 1

    # Hologram score
    patch = bgr[int(h * 0.55):int(h * 0.95), int(w * 0.05):int(w * 0.30)]
    if patch.size:
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hue_std = float(np.std(patch_hsv[:, :, 0]))
        sat_std = float(np.std(patch_hsv[:, :, 1]))
        holo_score = hue_std * 0.6 + sat_std * 0.4
    else:
        holo_score = 0.0

    # Raw OCR text (what tesseract actually reads, before regex)
    def _raw_ocr(crop):
        if crop.size == 0:
            return ""
        out = []
        for processed in forensic._preprocess_variants(crop):
            for psm in (7, 6, 8):
                out.append(forensic._ocr_region(processed, psm=psm))
        return " | ".join(t for t in out if t)

    ocr_top_left = _raw_ocr(
        bgr[int(h * 0.12):int(h * 0.32), int(w * 0.03):int(w * 0.32)]
    )
    ocr_bot_right = _raw_ocr(
        bgr[int(h * 0.78):h, int(w * 0.55):w]
    )

    return {
        "shape_hw": [h, w],
        "wm_variance": wm_variance,
        "uv_ratio_pct": uv_ratio * 100,
        "thread_vert_lines": vert_count,
        "hologram_score": holo_score,
        "ocr_top_left_raw": ocr_top_left[:300],
        "ocr_bottom_right_raw": ocr_bot_right[:300],
    }


# =====================================================
# WALK SAMPLES
# =====================================================

def collect():
    out = []
    for label in ("real", "fake"):
        d = os.path.join(SAMPLES_DIR, label)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            if not os.path.isfile(path):
                continue
            out.append((label, name, path))
    return out


def main():

    samples = collect()
    print(f"Running diagnostic on {len(samples)} samples ...\n")

    report = []

    # Per-sample summary table header
    print(
        f"{'LBL':<4} {'FILE':<32} {'MV':<6} {'RAW':>6} "
        f"{'F/7':>4} {'CMB':>6} {'VERDICT':<11} "
        f"{'WMVar':>8} {'UV%':>6} {'Thr':>4} {'Holo':>6}"
    )
    print("-" * 110)

    counts = {"real": {}, "fake": {}}

    for label, name, path in samples:

        t0 = time.time()
        full = run_full_pipeline(path)
        diag = diagnostic_signals(path)
        dt = time.time() - t0

        mv = "REAL" if full["raw"] >= 0.5 else "FAKE"

        line = (
            f"{label[:4]:<4} {name[:32]:<32} {mv:<6} "
            f"{full['raw']*100:>5.1f}% "
            f"{full['pass_count']}/{full['total']:<2} "
            f"{full['combined']*100:>5.1f}% "
            f"{full['verdict']:<11} "
            f"{diag['wm_variance']:>8.0f} "
            f"{diag['uv_ratio_pct']:>5.1f}% "
            f"{diag['thread_vert_lines']:>4} "
            f"{diag['hologram_score']:>6.1f}"
        )
        print(line)

        counts[label][full["verdict"]] = counts[label].get(
            full["verdict"], 0
        ) + 1

        report.append({
            "label": label,
            "file": name,
            "duration_s": round(dt, 2),
            "model_raw": full["raw"],
            "model_verdict": mv,
            "forensic_pass": full["pass_count"],
            "forensic_total": full["total"],
            "forensic_score": full["forensic_score"],
            "combined_score": full["combined"],
            "final_verdict": full["verdict"],
            "diagnostic": diag,
            "forensic_analysis": full["forensic_analysis"],
        })

    print("-" * 110)
    print("\nVERDICT COUNTS")
    for label in ("real", "fake"):
        print(f"  {label}:", counts[label])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
