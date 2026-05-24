"""Measure security-thread detection signals across every fixture
in the manifest. Used to validate the morphological vertical-line
detector before swapping out the current HoughLinesP implementation.

Production-grade approach: vertical Sobel + Otsu threshold +
morphological closing along a vertical kernel (connects windowed
dashes into continuous components) + opening to suppress noise +
connectedComponentsWithStats to count narrow tall components.
This is the standard CV technique for vertical-feature extraction
used in document analysis (table borders, barcode bars) and works
on both continuous and windowed threads."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


ROOT = pathlib.Path("tests/sample_notes")


def detect_thread_morphological(img: np.ndarray) -> dict:
    """Return {found, components, max_height_pct, max_density}."""

    h, w = img.shape[:2]
    if h < 60 or w < 60:
        return {
            "found": False,
            "components": 0,
            "max_height_pct": 0.0,
            "max_density": 0.0,
        }

    strip = img[0:h, int(w * 0.30):int(w * 0.70)]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # Vertical Sobel — picks up vertical edges
    sobel = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel)

    # Otsu binary threshold (adaptive to image brightness)
    _, binary = cv2.threshold(
        sobel_abs, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Vertical morphological closing: connect dashed segments
    # of windowed threads into continuous vertical features
    close_h = max(15, int(h * 0.08))
    vk_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vk_close)

    # Vertical opening: keep only column-aligned features that
    # extend at least 20% of the image height
    open_h = max(20, int(h * 0.20))
    vk_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_h))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vk_open)

    # Find connected components
    nlabels, _, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    components = 0
    max_height_pct = 0.0
    max_density = 0.0
    for i in range(1, nlabels):
        x, y, cw, ch, area = stats[i]
        height_pct = ch / h * 100.0
        # Density: fraction of the bounding box that is filled
        density = area / max(cw * ch, 1)
        if ch >= h * 0.25 and cw <= 15 and density >= 0.3:
            components += 1
            max_height_pct = max(max_height_pct, height_pct)
            max_density = max(max_density, density)

    return {
        "found": components >= 1,
        "components": components,
        "max_height_pct": max_height_pct,
        "max_density": max_density,
    }


def main():
    manifest = json.loads(
        (ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    samples = [(k, v) for k, v in manifest.items() if not k.startswith("_")]

    rows = []
    for k, v in samples:
        img = cv2.imread(str(ROOT / k))
        if img is None:
            continue
        m = detect_thread_morphological(img)
        rows.append((v["grade"], v.get("fake_type", ""), k, m))

    grade_order = {"clean": 0, "phone": 1, "edge_case": 2, "fake": 3}
    rows.sort(key=lambda r: (grade_order.get(r[0], 9), r[2]))

    print(f"{'grade':<10} {'subtype':<18} {'file':55} "
          f"{'found':>7} {'comps':>6} {'maxH%':>7} {'maxDens':>8}")
    print("-" * 120)
    for grade, sub, k, m in rows:
        print(
            f"{grade:<10} {sub:<18} {k:55} "
            f"{str(m['found']):>7} {m['components']:>6} "
            f"{m['max_height_pct']:>6.1f}% {m['max_density']:>8.2f}"
        )

    print("\n" + "=" * 120)
    print("PER-GRADE SEPARATION\n")
    by_grade = {}
    for grade, sub, k, m in rows:
        bucket = grade if grade != "fake" else f"fake/{sub}"
        by_grade.setdefault(bucket, []).append(m)

    print(f"{'bucket':<28}  {'found rate':>14}  {'avg components':>16}")
    print("-" * 80)
    for bucket, ms in by_grade.items():
        n = len(ms)
        found = sum(1 for m in ms if m["found"])
        avg_comp = sum(m["components"] for m in ms) / n
        print(f"{bucket:<28}  {found}/{n} ({found/n*100:>5.1f}%)  "
              f"{avg_comp:>15.2f}")


if __name__ == "__main__":
    main()
