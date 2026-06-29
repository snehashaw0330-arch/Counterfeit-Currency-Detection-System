"""
Fetch a representative slice of the JaalTaka Bangladeshi-banknote dataset
WITHOUT downloading the full 7.58 GB archive, into the LOCKED foreign layout.

JaalTaka (Mendeley Data, DOI 10.17632/2m7wk5cy4c.2, CC BY 4.0) ships as a single
monolithic `JaalTaka.zip` — 8,340 smartphone images of genuine and *physical*
counterfeit 500/1000 BDT notes, as `{real,fake}_notes/note_XXX/note_XXX_{1..6}.jpg`.
Each note is six high-resolution CLOSE-UP CROPS of different security regions
(portrait/serial, OVI numeral, hologram strip, watermark window, ...), NOT six
full-note photos — so they land under `security_crops` and feed the ML /
per-region side, never the whole-note geometric checks.

A ZIP's central directory lives at the end of the file, so we read it with HTTP
range requests and pull only a small, balanced, group-preserving slice (default
50 notes per class, evenly spaced across the note range).

Output — the locked layout validated by scripts/validate_foreign_dataset.py:
    dataset/foreign/bdt/security_crops/real/note_XXX_k.jpg  (+ .json sidecar)
    dataset/foreign/bdt/security_crops/fake/note_XXX_k.jpg  (+ .json sidecar)

Each sidecar carries: country, currency, substrate, denomination, side, label,
source (+ note_id / segment for group-aware splitting). The whole tree is
git-ignored and is kept OUT of the INR whole-note classifier
(scripts/build_dataset.py skips dataset/foreign/).

Requires: remotezip  (pip install remotezip)

Run:
    venv\\Scripts\\python.exe scripts\\fetch_jaaltaka.py [--per-class 50]
"""

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "foreign", "bdt", "security_crops")

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/2m7wk5cy4c/"
    "files/27e6e73f-407a-42ea-966c-1d669a3a41d9/file_downloaded"
)
SOURCE = "JaalTaka (Mendeley Data 10.17632/2m7wk5cy4c.2, CC BY 4.0)"


def _resolve_s3(url):
    """Resolve the Mendeley endpoint's 302 to its public S3 object (which honours
    range requests). curl is used because Cloudflare bot-blocks python-requests'
    User-Agent; curl ships with Windows 10+, macOS and Linux."""
    s3 = subprocess.check_output(
        ["curl", "-sS", "-o", os.devnull, "-w", "%{redirect_url}", url],
        timeout=120,
    ).decode().strip()
    if not s3.startswith("http"):
        raise RuntimeError(f"could not resolve download redirect (got: {s3!r})")
    return s3


def _evenly_spaced(items, n):
    """Pick n items spread across the full range (deterministic)."""
    if n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def _note_id(name):
    # "real_notes/note_017/note_017_3.jpg" -> "note_017"
    return name.split("/")[1]


def _segment(name):
    m = re.search(r"_(\d+)\.jpg$", name, re.I)
    return m.group(1) if m else ""


def _sidecar(label, note_id, segment):
    return {
        "country": "Bangladesh",
        "currency": "BDT",
        "substrate": "paper",          # BDT 500/1000 are paper, not polymer
        "denomination": "unknown",     # JaalTaka mixes 500/1000; not per-note labelled
        "side": "crop",
        "label": label,                # "real" | "fake"
        "source": SOURCE,
        "note_id": note_id,            # extra: for group-aware train/test splits
        "segment": segment,            # extra: which of the 6 security regions
    }


def fetch(per_class):
    try:
        from remotezip import RemoteZip
    except ImportError:
        print("ERROR: remotezip is required. Install it with:\n"
              "    venv\\Scripts\\python.exe -m pip install remotezip")
        return 1

    print("Resolving download URL ...")
    s3 = _resolve_s3(MENDELEY_URL)
    print(f"  -> {s3[:80]}...")

    print("Reading archive index via HTTP range (no full download) ...")
    written = 0
    with RemoteZip(s3) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
        real = sorted(n for n in names if n.startswith("real_notes/"))
        fake = sorted(n for n in names if n.startswith("fake_notes/"))

        for src_prefix, label, members in (
            ("real_notes", "real", real),
            ("fake_notes", "fake", fake),
        ):
            note_ids = sorted({_note_id(n) for n in members})
            picked = set(_evenly_spaced(note_ids, per_class))
            todo = [n for n in members if _note_id(n) in picked]
            dest_dir = os.path.join(OUT_DIR, label)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"\n{label}: {len(picked)} notes / {len(todo)} images "
                  f"(of {len(note_ids)} available)")

            for i, name in enumerate(todo, 1):
                base = os.path.basename(name)              # note_XXX_k.jpg
                img_path = os.path.join(dest_dir, base)
                meta_path = os.path.splitext(img_path)[0] + ".json"
                if (os.path.exists(img_path) and os.path.getsize(img_path) > 0
                        and os.path.exists(meta_path)):
                    continue  # idempotent
                with open(img_path, "wb") as fh:
                    fh.write(z.read(name))
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(_sidecar(label, _note_id(name), _segment(name)),
                              fh, indent=2)
                written += 1
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} ...")

    print(f"\nDone. Wrote {written} new image(s)+sidecar(s) under "
          f"{os.path.relpath(OUT_DIR, ROOT)}")
    print("Validate: venv\\Scripts\\python.exe scripts\\validate_foreign_dataset.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=50,
                    help="notes per class to fetch (each note = 6 crops). "
                         "Default 50 -> ~600 images, ~390 MB.")
    args = ap.parse_args()
    sys.exit(fetch(args.per_class))
