"""Smoke-test EasyOCR-based serial + denom on every phone fixture in the manifest."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2

from backend.forensic import (
    classify_denomination,
    extract_serial_number,
    warmup_ocr,
)


def main():
    warmup_ocr()

    root = pathlib.Path("tests/sample_notes")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    print(f"{'file':55} {'expect denom':>12} {'got denom':>10}  {'expect serial':>14}  got serial")
    print("-" * 130)

    for k, v in manifest.items():
        if k.startswith("_"):
            continue
        if v["grade"] != "phone":
            continue

        img = cv2.imread(str(root / k))
        if img is None:
            print(f"{k:55} COULD NOT READ")
            continue

        got_denom = classify_denomination(img)
        got_serial = extract_serial_number(img)

        denom_match = "OK " if str(got_denom.get("value")) == str(v["denom"]) else "BAD"
        serial_match = "OK " if (
            v["serial"] is None
            or got_serial.get("value") == v["serial"]
        ) else "BAD"

        print(
            f"{k:55} "
            f"{str(v['denom']):>12} "
            f"{str(got_denom.get('value')):>10} {denom_match}  "
            f"{str(v['serial']):>14}  "
            f"{str(got_serial.get('value')):>20} {serial_match}"
        )


if __name__ == "__main__":
    main()
