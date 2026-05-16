"""Run the full /predict-equivalent verdict on every phone fixture.

Replicates main.py's verdict combiner logic without spinning up FastAPI,
so we can check whether real notes now produce REAL verdicts (not FAKE).
"""

import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from backend.forensic import run_forensic_pipeline, warmup_ocr


def combine_verdict(fa: dict) -> tuple[str, float, int, int]:
    scored = [
        c for k, c in fa.items()
        if k != "modular_ai_pipeline" and c["status"] in ("PASS", "FAIL")
    ]
    pass_count = sum(1 for c in scored if c["status"] == "PASS")
    total = max(len(scored), 1)
    forensic_score = pass_count / total

    sanity = fa.get("structural_sanity", {})
    colour = fa.get("hologram_detection", {})

    if sanity.get("status") == "FAIL":
        return "FAKE", forensic_score, pass_count, total

    # Without the ML model output, approximate combined = forensic_score
    # (we're not running the keras model here to keep this fast)
    if forensic_score >= 0.65 and pass_count >= 5 and colour.get("status") != "FAIL":
        return "REAL", forensic_score, pass_count, total
    if forensic_score < 0.35:
        return "FAKE", forensic_score, pass_count, total
    return "SUSPICIOUS", forensic_score, pass_count, total


def main():
    warmup_ocr()

    root = pathlib.Path("tests/sample_notes")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    print(f"{'file':55} {'denom':>5} {'verdict':>11} {'forensic':>10} {'pass'}")
    print("-" * 110)

    for k, v in manifest.items():
        if k.startswith("_") or v["grade"] != "phone":
            continue

        img = cv2.imread(str(root / k))
        if img is None:
            continue

        fa = run_forensic_pipeline(img)
        verdict, fscore, passc, total = combine_verdict(fa)

        denom = fa.get("denomination_classification", {}).get("value")
        struct = fa.get("structural_sanity", {})
        struct_short = f"{struct.get('status', '?')[:4]}"

        print(
            f"{k:55} "
            f"{str(denom):>5} "
            f"{verdict:>11} "
            f"{fscore*100:>8.1f}% "
            f"{passc}/{total}  "
            f"struct={struct_short} ({struct.get('details', '')[:50]})"
        )


if __name__ == "__main__":
    main()
