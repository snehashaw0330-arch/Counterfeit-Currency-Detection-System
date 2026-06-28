# BankNote-Net

This project now has a clean path for adding a **multi-country currency
dataset** without polluting the current INR genuine-vs-fake image benchmark.

## What it is

BankNote-Net is a public embeddings dataset from Microsoft for assistive
currency recognition.

- 24k+ examples
- 17 currencies
- 112 denominations
- 256-dimensional learned banknote embeddings

It is useful for:

- country detection
- currency classification
- denomination routing
- few-shot foreign-note experiments

It is **not** a drop-in replacement for our current image-based counterfeit
classifier, because it ships embeddings rather than raw note photos.

## Repo workflow

Download the dataset:

```powershell
venv\Scripts\python.exe scripts\fetch_banknote_net.py
```

Train a multi-currency classifier on top of the embeddings:

```powershell
venv\Scripts\python.exe scripts\train_banknote_net_currency.py
```

Outputs:

- `dataset/foreign/banknote_net/banknote_net.csv`
- `dataset/foreign/banknote_net/manifest.json`
- `models/banknote_net/metrics.json`

## Why we keep it separate

Our existing pipeline:

- `scripts/build_dataset.py`
- `scripts/train_classical.py`

expects **raw images** labelled as `real` or `fake`.

BankNote-Net should therefore be treated as a separate branch for:

- automatic country detection
- multi-currency routing
- pre-routing before country-specific counterfeit checks

and **not** merged into the INR whole-note binary classifier.

## Recommended use in this project

1. Use **JaalTaka (Bangladesh)** for foreign counterfeit coverage.
2. Use **BankNote-Net** for broad multi-country country/currency detection.
3. Route a note to the right country-specific forensic pipeline after country
   classification.

## Related foreign modules

The live foreign/polymer work in this repo is intentionally separate from
BankNote-Net:

- `backend/country.py`:
  OCR-first live country/currency routing with an explicit unknown fallback.
- `backend/polymer.py`:
  conservative polymer cues (`PASS` or `INFO`, never substrate-only `FAIL`).
- `scripts/benchmark_foreign_modules.py`:
  per-currency local benchmark over `dataset/foreign/`.
- `scripts/validate_foreign_dataset.py`:
  layout + sidecar-metadata validation for the locked foreign dataset structure.

## Source

- GitHub repository:
  https://github.com/microsoft/banknote-net
- Paper:
  https://arxiv.org/abs/2204.03738
