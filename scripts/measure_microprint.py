"""
Calibration probe for analyze_microprint (Phase E).

Computes _microprint_score over every image in dataset/index.json
and prints the genuine-vs-fake distribution so the PASS/FAIL
thresholds in backend/forensic.py can be set from evidence rather
than guessed. Mirrors the existing scripts/measure_*.py probes.

Run:
    venv\\Scripts\\python.exe scripts\\measure_microprint.py
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.forensic import _microprint_score, _locate_note  # noqa: E402

INDEX = os.path.join(ROOT, "dataset", "index.json")


def _stats(label, scores):
    if not scores:
        print(f"  {label}: (none)")
        return
    a = np.array(scores)
    print(f"  {label:18} n={len(a):3}  "
          f"min={a.min():7.1f}  p25={np.percentile(a,25):7.1f}  "
          f"median={np.median(a):7.1f}  p75={np.percentile(a,75):7.1f}  "
          f"max={a.max():8.1f}")


def main():
    with open(INDEX, "r", encoding="utf-8") as fh:
        samples = json.load(fh)["samples"]

    by_label = {"genuine": [], "fake": []}
    rows = []

    for s in samples:
        img = cv2.imread(os.path.join(ROOT, s["path"].replace("/", os.sep)))
        if img is None:
            continue
        # Measure on the located note, matching the live pipeline.
        note = _locate_note(img)
        score, native_w = _microprint_score(note)
        by_label[s["label"]].append(score)
        rows.append((s["label"], score, native_w, s["path"]))

    print("=== micro-print sharpness distribution ===")
    _stats("genuine", by_label["genuine"])
    _stats("fake", by_label["fake"])

    print("\n=== per-image (sorted by score) ===")
    for label, score, nw, path in sorted(rows, key=lambda r: r[1]):
        tag = "FAKE" if label == "fake" else "real"
        print(f"  {tag:4} {score:8.1f}  w={nw:5}  {path}")


if __name__ == "__main__":
    raise SystemExit(main())
