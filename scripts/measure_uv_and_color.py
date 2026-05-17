"""Measure raw UV ratio, mean saturation, and hue entropy across every
fixture in the manifest. Used to retune the analyze_uv_features and
detect_hologram thresholds against the full corpus instead of the
original 5-fixture sample."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


ROOT = pathlib.Path("tests/sample_notes")


def measure(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # UV proxy (current analyze_uv_features formula)
    high_sat = cv2.inRange(hsv, (0, 120, 120), (179, 255, 255))
    uv_ratio = float(np.count_nonzero(high_sat)) / high_sat.size

    # Colour palette signals (current detect_hologram formula)
    sat = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    mean_sat = float(np.mean(sat))

    coloured = hue[sat > 60]
    if coloured.size < 200:
        hue_entropy = 0.0
    else:
        hist = np.bincount(coloured, minlength=180).astype(np.float64)
        hist /= hist.sum()
        nz = hist[hist > 0]
        hue_entropy = float(-(nz * np.log2(nz)).sum())

    return {
        "uv_pct": uv_ratio * 100,
        "mean_sat": mean_sat,
        "hue_entropy": hue_entropy,
    }


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    samples = [(k, v) for k, v in manifest.items() if not k.startswith("_")]

    rows = []
    for k, v in samples:
        img = cv2.imread(str(ROOT / k))
        if img is None:
            continue
        m = measure(img)
        rows.append((v["grade"], v.get("fake_type", ""), k, m))

    grade_order = {"clean": 0, "phone": 1, "edge_case": 2, "fake": 3}
    rows.sort(key=lambda r: (grade_order.get(r[0], 9), r[2]))

    print(f"{'grade':<10} {'subtype':<18} {'file':55} {'UV%':>7} {'mean_sat':>10} {'hue_entropy':>12}")
    print("-" * 120)
    for grade, sub, k, m in rows:
        print(
            f"{grade:<10} {sub:<18} {k:55} "
            f"{m['uv_pct']:>6.2f}%  {m['mean_sat']:>9.1f}  {m['hue_entropy']:>11.2f}"
        )

    # Per-grade summary
    print("\n" + "=" * 120)
    print("PER-GRADE SUMMARY (min / median / max)\n")
    by_grade = {}
    for grade, sub, k, m in rows:
        bucket = grade if grade != "fake" else f"fake/{sub}"
        by_grade.setdefault(bucket, []).append(m)

    print(f"{'bucket':<28} {'UV%':>25} {'mean_sat':>25} {'hue_entropy':>25}")
    print("-" * 120)

    def stats(values):
        vals = sorted(values)
        return vals[0], vals[len(vals) // 2], vals[-1]

    for bucket, ms in by_grade.items():
        uv_min, uv_med, uv_max = stats([m["uv_pct"] for m in ms])
        sat_min, sat_med, sat_max = stats([m["mean_sat"] for m in ms])
        he_min, he_med, he_max = stats([m["hue_entropy"] for m in ms])
        print(
            f"{bucket:<28} "
            f"{uv_min:>5.2f} / {uv_med:>5.2f} / {uv_max:>5.2f}  "
            f"{sat_min:>6.1f} / {sat_med:>6.1f} / {sat_max:>6.1f}  "
            f"{he_min:>6.2f} / {he_med:>6.2f} / {he_max:>6.2f}"
        )


if __name__ == "__main__":
    main()
