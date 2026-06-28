"""
Benchmark the self-contained foreign modules on the local foreign dataset tree.

Evaluates:
- backend.country.detect_country on full-note real images with sidecar metadata
- backend.polymer.analyze_polymer_features on polymer-labelled real images

Run:
    venv\\Scripts\\python.exe scripts\\benchmark_foreign_modules.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.country import detect_country  # noqa: E402
from backend.polymer import analyze_polymer_features  # noqa: E402


BASE = os.path.join(ROOT, "dataset", "foreign")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_samples():
    if not os.path.isdir(BASE):
        return
    for root, _dirs, files in os.walk(BASE):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXT:
                continue
            img_path = os.path.join(root, name)
            meta_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            yield img_path, meta


def main() -> int:
    total = 0
    correct = 0
    per_currency = {}
    polymer_total = 0
    polymer_pass = 0

    for img_path, meta in _iter_samples() or []:
        if meta.get("label") != "real" or "full_note" not in img_path.replace("\\", "/"):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        total += 1
        pred = detect_country(img)
        gt = meta.get("currency")
        bucket = per_currency.setdefault(gt, {"n": 0, "ok": 0})
        bucket["n"] += 1
        if pred.get("currency") == gt:
            correct += 1
            bucket["ok"] += 1

        if meta.get("substrate") == "polymer":
            polymer_total += 1
            poly = analyze_polymer_features(img)
            if poly.get("status") == "PASS":
                polymer_pass += 1

    print(f"Country detection: {correct}/{total} "
          f"({(correct / total * 100.0) if total else 0.0:.1f}%)")
    for ccy in sorted(per_currency):
        row = per_currency[ccy]
        pct = (row["ok"] / row["n"] * 100.0) if row["n"] else 0.0
        print(f"  {ccy}: {row['ok']}/{row['n']} ({pct:.1f}%)")

    if polymer_total:
        pct = polymer_pass / polymer_total * 100.0
        print(f"Polymer cue PASS coverage: {polymer_pass}/{polymer_total} ({pct:.1f}%)")
    else:
        print("Polymer cue PASS coverage: no polymer-labelled real samples found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
