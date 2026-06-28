"""
Download the public BankNote-Net multi-currency embeddings dataset.

Why this exists
---------------
Our current `build_dataset.py` / `train_classical.py` pipeline expects raw note
images and is tuned for INR genuine-vs-fake classification. For multi-country
work we also need a broader currency dataset that does NOT get mixed into that
binary image classifier by accident.

BankNote-Net is a good fit for the new country/currency-detection branch:

- 24k+ banknote embeddings
- 17 currencies
- public GitHub-hosted CSV
- no extra Kaggle login / manual download step

This script downloads the CSV plus a small summary manifest under:

    dataset/foreign/banknote_net/
        banknote_net.csv
        manifest.json

Run:
    venv\\Scripts\\python.exe scripts\\fetch_banknote_net.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "foreign", "banknote_net")
CSV_URL = (
    "https://raw.githubusercontent.com/microsoft/banknote-net/main/data/"
    "banknote_net.csv"
)
CSV_PATH = os.path.join(OUT_DIR, "banknote_net.csv")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def _find_col(header: list[str], wanted: str) -> str | None:
    wanted = wanted.lower()
    for col in header:
        if col.strip().lower() == wanted:
            return col
    for col in header:
        if wanted in col.strip().lower():
            return col
    return None


def _build_manifest(csv_path: str) -> dict:
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []

        currency_col = _find_col(header, "currency")
        denomination_col = _find_col(header, "denomination")
        face_col = _find_col(header, "face")

        if not currency_col:
            raise RuntimeError(
                f"Could not find a currency column in {csv_path}. "
                f"Header starts with: {header[:8]}"
            )

        numeric_like = []
        for col in header:
            low = col.strip().lower()
            if low in {"currency", "denomination", "face"}:
                continue
            if low.startswith("v_"):
                numeric_like.append(col)
                continue
            if low.startswith("emb") or low.startswith("feat"):
                numeric_like.append(col)
                continue
            try:
                float(low)
                numeric_like.append(col)
            except ValueError:
                pass

        n_rows = 0
        by_currency: Counter[str] = Counter()
        by_denom: Counter[str] = Counter()
        by_face: Counter[str] = Counter()

        for row in reader:
            n_rows += 1
            cur = str(row.get(currency_col, "")).strip()
            if cur:
                by_currency[cur] += 1
            if denomination_col:
                den = str(row.get(denomination_col, "")).strip()
                if den:
                    by_denom[f"{cur}:{den}"] += 1
            if face_col:
                face = str(row.get(face_col, "")).strip()
                if face:
                    by_face[face] += 1

    top_denoms = [
        {"currency_and_denomination": k, "count": v}
        for k, v in by_denom.most_common(20)
    ]

    return {
        "source": {
            "name": "BankNote-Net",
            "url": CSV_URL,
            "repo": "https://github.com/microsoft/banknote-net",
            "license": "CDLA-Permissive-2.0",
        },
        "shape": {
            "rows": n_rows,
            "columns_total": len(header),
            "embedding_columns_detected": len(numeric_like),
        },
        "schema": {
            "currency_column": currency_col,
            "denomination_column": denomination_col,
            "face_column": face_col,
        },
        "counts": {
            "currencies": dict(sorted(by_currency.items())),
            "faces": dict(sorted(by_face.items())),
            "top_currency_denominations": top_denoms,
        },
        "notes": [
            "This is an embeddings dataset, not raw note photos.",
            "Keep it separate from dataset/index.json and the INR genuine/fake image classifier.",
            "Use it for country/currency detection or few-shot foreign-note experiments.",
        ],
    }


def main() -> int:
    print("Downloading BankNote-Net CSV ...")
    _download(CSV_URL, CSV_PATH)
    size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
    print(f"  saved {os.path.relpath(CSV_PATH, ROOT)} ({size_mb:.1f} MB)")

    manifest = _build_manifest(CSV_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    counts = manifest["counts"]["currencies"]
    print(f"  rows: {manifest['shape']['rows']}")
    print(f"  currencies: {len(counts)} -> {', '.join(sorted(counts))}")
    print(f"  wrote {os.path.relpath(MANIFEST_PATH, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
