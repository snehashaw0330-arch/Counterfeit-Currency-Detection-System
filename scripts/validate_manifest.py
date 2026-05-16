"""Validate tests/sample_notes/manifest.json: parses, all files exist, print tallies."""

import json
import pathlib
import sys

ROOT = pathlib.Path("tests/sample_notes")
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

samples = {k: v for k, v in manifest.items() if not k.startswith("_")}
missing = [k for k in samples if not (ROOT / k).exists()]

print(f"manifest entries: {len(samples)}")
print(f"missing on disk:  {missing if missing else 'none'}")

grades = ["clean", "phone", "edge_case", "fake"]
tally = {g: sum(1 for v in samples.values() if v["grade"] == g) for g in grades}
print(f"grade tally:      {tally}")

phone_with_serial = [
    (k, v["serial"]) for k, v in samples.items()
    if v["grade"] == "phone" and v["serial"] is not None
]
print(f"phone known-serial samples ({len(phone_with_serial)}):")
for k, s in phone_with_serial:
    print(f"  {s:>14}  {k}")

if missing:
    sys.exit(1)
