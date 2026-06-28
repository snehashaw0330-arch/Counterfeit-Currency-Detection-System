"""
Fetch a representative slice of the JaalTaka Bangladeshi-banknote dataset
WITHOUT downloading the full 7.58 GB archive.

JaalTaka (Mendeley Data, DOI 10.17632/2m7wk5cy4c.2, CC BY 4.0) ships as a single
monolithic `JaalTaka.zip` — 8,340 smartphone images of genuine and *physical*
counterfeit 500/1000 BDT notes, laid out as:

    real_notes/note_XXX/note_XXX_{1..6}.jpg   (802 notes -> 4,812 images)
    fake_notes/note_XXX/note_XXX_{1..6}.jpg   (588 notes -> 3,528 images)

Each note has 6 high-resolution CLOSE-UP CROPS of different security regions
(portrait/serial, OVI numeral, hologram strip, watermark window, ...), NOT six
full-note photos. They are therefore used by the ML / per-region forensic side,
not the whole-note geometric checks (locate / proportions / full serial).

We pull only a small, balanced, group-preserving slice (default 50 notes per
class, evenly spaced across the note range so both denominations + varied
conditions are represented). A ZIP's central directory lives at the end of the
file, so we read it with HTTP range requests and then range-fetch only the
members we want — a few hundred MB instead of 7.58 GB.

Output layout (git-ignored, picked up by scripts/build_dataset.py):
    dataset/foreign/bdt/real_notes/note_XXX/note_XXX_k.jpg   -> genuine
    dataset/foreign/bdt/fake_notes/note_XXX/note_XXX_k.jpg   -> fake

NOTE: do NOT re-run train_classical.py after this until the loader is made
country-aware (Phase S.1) — otherwise these BDT crops get pooled with the INR
full-note corpus and contaminate the committed Indian benchmark.

Requires: remotezip  (pip install remotezip)

Run:
    venv\\Scripts\\python.exe scripts\\fetch_jaaltaka.py [--per-class 50]
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "foreign", "bdt")

# Durable public download URL for JaalTaka.zip (the Mendeley file endpoint
# 302-redirects to a public S3 object that honours HTTP range requests).
MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/2m7wk5cy4c/"
    "files/27e6e73f-407a-42ea-966c-1d669a3a41d9/file_downloaded"
)


def _resolve_s3(url):
    """Return the S3 URL the Mendeley endpoint redirects to.

    Cloudflare bot-blocks python-requests' default User-Agent (no 302), but
    lets curl through; the underlying S3 object serves any client and honours
    range requests. curl ships with Windows 10+, macOS and Linux."""
    s3 = subprocess.check_output(
        ["curl", "-sS", "-o", os.devnull, "-w", "%{redirect_url}", url],
        timeout=120,
    ).decode().strip()
    if not s3.startswith("http"):
        raise RuntimeError(f"could not resolve download redirect (got: {s3!r})")
    return s3


def _evenly_spaced(items, n):
    """Pick n items spread across the full range (not just the first n), so a
    slice spans both denominations / conditions. Deterministic."""
    if n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def _note_id(name):
    # "real_notes/note_017/note_017_3.jpg" -> "note_017"
    return name.split("/")[1]


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

        for cls, members in (("real_notes", real), ("fake_notes", fake)):
            note_ids = sorted({_note_id(n) for n in members})
            picked = set(_evenly_spaced(note_ids, per_class))
            todo = [n for n in members if _note_id(n) in picked]
            print(f"\n{cls}: {len(picked)} notes / {len(todo)} images "
                  f"(of {len(note_ids)} notes available)")

            for i, name in enumerate(todo, 1):
                dest = os.path.join(OUT_DIR, name.replace("/", os.sep))
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    continue  # idempotent: skip already-fetched
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(z.read(name))
                written += 1
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} ...")

    print(f"\nDone. Wrote {written} new image(s) under "
          f"{os.path.relpath(OUT_DIR, ROOT)}")
    print("Next: build the country-aware index (Phase S.1), then benchmark "
          "per-currency. Do NOT pool these crops with the INR corpus.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=50,
                    help="notes per class to fetch (each note = 6 images). "
                         "Default 50 -> ~600 images, ~390 MB.")
    args = ap.parse_args()
    sys.exit(fetch(args.per_class))
