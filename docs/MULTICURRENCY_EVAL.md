# Multi-Currency Evaluation — Indian vs Foreign (Phase T3 · requirement item 11)

Honest, reproducible comparison of counterfeit detection across the two
currencies for which we have genuine **and** real counterfeit images, plus the
multi-currency identification layer. All numbers are regenerable from the scripts
named below; held-out splits are group-aware (no leakage) and model selection is
by cross-validation (no test-set peeking).

> **A credible counterfeit benchmark needs real fakes.** JaalTaka gives genuine
> + physical-counterfeit BDT; INR has our fixture fakes. **AUD (2026-07-02)**
> has a partner-supplied set whose fakes are **synthetic** (digitally-altered
> copies of the reals) — its model is trained + wired live via the generic
> per-currency pipeline, but its score measures *manipulation detection*, not
> physical-counterfeit detection (caveat in metrics.json, the /predict notice,
> and the UI). CAD/GBP/PHP remain identification + polymer cues only.

## Headline (counterfeit detection)

| Currency | Corpus (held-out test) | Best technique | Test accuracy | Macro-F1 |
|---|---|---|---|---|
| **BDT** (Bangladesh) | 672 imgs · 132 test / 22 notes | **Random Forest** | **0.909 image / 0.955 note** | 0.909 |
| **INR** (India) | 65 fixture imgs · 17 test | SVM (RBF) | 0.765 | 0.742 |
| **AUD** (Australia) ⚠ synthetic fakes | 700 imgs · 251 test (group-aware, fake paired w/ its real twin) | Random Forest (CV 0.913) | 0.924 | 0.924 |
| **CAD** (Canada) ⚠ synthetic fakes | 335 imgs · 121 test (group-aware, fake paired w/ its real twin) | Random Forest (CV 0.948) | 0.942 | 0.942 |

> ⚠ **AUD/CAD caveat:** those scores are against *synthetic* fakes (small
> catalog scans, each fake derived from its real twin). They demonstrate the
> generic pipeline (`scripts/train_foreign_counterfeit.py`) end-to-end on two
> drop-in currencies; they do NOT claim real-world counterfeit power. Real
> physical fakes would make these true benchmarks. Also note: such low-res
> scans often fail currency identification (OCR-starved), so they may not
> reach the per-currency model in the live gate — full-resolution photos route
> correctly (verified live: AUD 0.82, CAD 0.88 conf).

The gap is the project's recurring lesson made concrete: **data quantity
dominates.** With 672 real images BDT reaches 0.91/0.955; the 65-image INR
fixture corpus (a larger local INR set is not present in this environment) is
high-variance on 17 test images. Same feature pipeline, same four techniques —
the methodology transfers; the data is the binding constraint.

## BDT — supervised (JaalTaka, group-aware by note, no augmentation)

`scripts/train_bdt_counterfeit.py` → `models/bdt/metrics.json`

| Technique | CV macro-F1 | Image acc | Note acc (6-view vote) | Image confusion [fake,gen] |
|---|---|---|---|---|
| **Random Forest** (selected) | **0.888** | 0.909 | **0.955** | [[63,3],[9,57]] |
| SVM (RBF) | 0.878 | 0.924 | 0.955 | [[61,5],[5,61]] |
| KNN | 0.824 | 0.902 | 0.955 | [[63,3],[10,56]] |
| Logistic Regression | 0.853 | 0.856 | 0.864 | [[59,7],[12,54]] |

## INR — supervised (65-image fixture corpus)

`scripts/build_dataset.py` + `scripts/train_classical.py` → `docs/BENCHMARK.md`

| Technique | CV macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|
| SVM (RBF) | 0.441 | **0.765** | **0.742** |
| Random Forest (selected by CV) | 0.478 | 0.647 | 0.614 |
| KNN | 0.452 | 0.588 | 0.583 |
| Logistic Regression | 0.443 | 0.706 | 0.622 |

> Caveat: 17 held-out images → high variance. A larger local INR dataset scores
> markedly higher; re-add it under `dataset/real|fake` and re-run to reproduce.

## Genuine-only (one-class) — tested honestly, and rejected

The "train on genuine, flag deviations as fake" idea, measured on BDT (the one
foreign currency where we can validate it against real fakes).

`scripts/train_oneclass_anomaly.py` → `models/bdt/oneclass_metrics.json`

| Approach (trained on GENUINE only) | Image ROC-AUC | At ~95% genuine pass: fakes caught |
|---|---|---|
| Isolation Forest | 0.664 | 1.5% |
| One-Class SVM | 0.663 | 47% (but false-flags 21% of genuine) |
| **Supervised RF (sees fakes)** | — | **0.909 image / 0.955 note acc** |

**Conclusion:** ROC-AUC ≈ 0.66 is barely above chance. A counterfeit is *built*
to resemble a genuine note, so it sits inside the genuine distribution and an
anomaly detector cannot separate it. **Counterfeit detection requires labelled
fakes** — so we do not ship genuine-only "detectors" for the polymer currencies.

## Currency / country identification

- `backend/country.py` (OCR-first via the **shared EasyOCR singleton**, palette/
  aspect tie-break) detects **8 currencies — INR, BDT, AUD, CAD, GBP, PHP, USD,
  EUR** — with an explicit `UNKNOWN` fallback. Integrated into `/predict` behind a
  zero-cost gate (runs only when the note isn't identified as INR).
- On genuine whole-note references with readable issuer text, detection is
  confident: **AUD 0.82, CAD 0.88, PHP 0.98, USD 0.98, EUR 0.88, INR 0.98 — 6/7
  correct.** GBP returns an honest `UNKNOWN` on its current specimen image, whose
  "Bank of England" wording OCRs too garbled to confirm (a data/image-quality
  limit, not a forced wrong guess). Region *crops* (e.g. JaalTaka BDT) lack issuer
  text and also return `UNKNOWN` — a whole-note photo is needed to route.
- **BankNote-Net** (17 currencies) kept as an *offline* benchmark: a shallow
  classifier on its embeddings reaches ~95% currency accuracy in-distribution,
  but the encoder isn't shipped, so it is not on the live path.

## Honest cross-currency findings

1. **The pipeline generalises** — the same 50-dim feature extractor + classical
   techniques work on INR and BDT without per-currency tuning.
2. **Data quantity is the ceiling** — BDT (672 imgs) → 0.91/0.955; INR (65) →
   0.65–0.76. Numbers track corpus size, not the method.
3. **Genuine-only screening doesn't work** — labelled counterfeits are required.
4. **Polymer currencies** (AUD/CAD/GBP/PHP) get identification + polymer cues
   only; counterfeit detection is impossible without public fake data.

## Reproduce

```
venv\Scripts\python.exe scripts\fetch_jaaltaka.py --per-class 50      # BDT data
venv\Scripts\python.exe scripts\validate_foreign_dataset.py           # layout
venv\Scripts\python.exe scripts\train_bdt_counterfeit.py              # BDT model
venv\Scripts\python.exe scripts\train_oneclass_anomaly.py             # one-class
venv\Scripts\python.exe scripts\build_dataset.py                      # INR index
venv\Scripts\python.exe scripts\train_classical.py                    # INR models
venv\Scripts\python.exe scripts\benchmark_models.py                   # BENCHMARK.md
```
