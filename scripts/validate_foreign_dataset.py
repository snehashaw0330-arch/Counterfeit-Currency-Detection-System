"""
Validate the locked foreign dataset layout and required sidecar metadata.

Expected image layout:
    dataset/foreign/<ccy>/{full_note,security_crops}/{real,fake}/<file>

Expected sidecar JSON keys:
    country, currency, substrate, denomination, side, label, source
"""

from __future__ import annotations

import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset", "foreign")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_KEYS = {
    "country", "currency", "substrate", "denomination", "side", "label", "source",
}


def main() -> int:
    bad = 0
    checked = 0
    if not os.path.isdir(BASE):
        print("dataset/foreign/ does not exist yet")
        return 1

    for root, _dirs, files in os.walk(BASE):
        rel = os.path.relpath(root, BASE).replace("\\", "/")
        parts = [p for p in rel.split("/") if p and p != "."]
        if len(parts) >= 3:
            split_ok = parts[1] in {"full_note", "security_crops"} and parts[2] in {"real", "fake"}
            if not split_ok:
                print(f"BAD layout folder: {rel}")
                bad += 1

        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXT:
                continue
            checked += 1
            img_path = os.path.join(root, name)
            meta_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.exists(meta_path):
                print(f"Missing metadata: {os.path.relpath(meta_path, ROOT)}")
                bad += 1
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception as exc:
                print(f"Unreadable metadata {os.path.relpath(meta_path, ROOT)}: {exc}")
                bad += 1
                continue
            missing = sorted(REQUIRED_KEYS - set(meta))
            if missing:
                print(f"Missing keys in {os.path.relpath(meta_path, ROOT)}: {missing}")
                bad += 1

    print(f"Checked {checked} foreign image(s)")
    if bad:
        print(f"Validation failed with {bad} issue(s)")
        return 1
    print("Foreign dataset layout looks good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
