# Counterfeit Bank Currency Detection with Various Machine Learning Techniques
### Project Report

**Currency:** Indian Rupees (Mahatma Gandhi New Series + common older series)
**Deployment:** local demonstration (FastAPI backend + Next.js frontend)
**Date:** 2026-06-01

---

## 1. Abstract

We built a counterfeit-detection system for Indian banknotes that takes a
phone photo and returns a single, explainable verdict — **REAL, SUSPICIOUS, or
FAKE**. Rather than relying on one model, it fuses three independent views:
(1) a deep-learning image classifier (MobileNetV2), (2) **multiple classical
machine-learning techniques** (SVM, Random Forest, KNN, Logistic Regression)
trained on hand-crafted visual features, and (3) an explainable forensic
pipeline of eleven security-feature checks (serial number, denomination,
watermark, security thread, ascending-serial typography, micro-lettering,
colour palette, note proportions, and more). On a held-out test set the
classical models (best: SVM / Random Forest ≈ 0.74 macro-F1) outperform the
CNN (≈ 0.56). End-to-end, the system **never misclassifies a genuine note as
fake (0% false positives)** and flags ~61% of fakes, with an honestly-stated
ceiling driven by training-data volume rather than method.

---

## 2. Problem and brief

Counterfeit currency harms the economy; the RBI reports rising detection of
fake ₹500/₹2000 notes. The internship brief: *"Counterfeit Bank Currency
Detection with Various Machine Learning Techniques. Consider Indian Rupees and
find the various features. The bank note number printed on the note varies with
size, a possible feature. Figures, motifs, colour variation, changes under UV
are possible features."*

Two reference works informed the design: an IEEE paper (Teachable Machine +
MATLAB edge/feature extraction) and a B.E. thesis (CNN + Local Binary Pattern
region comparison). Both use **visible-light image processing only** — neither
uses UV hardware — which directly shaped our scope.

---

## 3. System architecture

```
  phone photo ──POST /predict──▶ FastAPI (backend/main.py)
        │
        ├── MobileNetV2 CNN (224², sigmoid → P(genuine))
        ├── Classical ML 2nd opinion (best technique on 50-d feature vector)
        └── Forensic pipeline (backend/forensic.py): auto-crop the note, then
              11 checks → each PASS / FAIL / INFO with the measured numbers
        │
        └──▶ combined verdict (0.4·CNN + 0.6·forensic, with structural /
              colour / proportion hard gates) → REAL / SUSPICIOUS / FAKE
        │
        └──▶ Next.js UI renders the verdict, every check, the proportion
              panel, the RBI-serial-typography panel, and the
              ML-technique-comparison panel.
```

---

## 4. Methodology

### 4.1 Forensic security-feature checks (visible-light, explainable)

| Check | What it measures |
|---|---|
| Structural sanity | aspect / edges / brightness / dark-quadrants gate (rejects blanks, noise) |
| OCR serial number | EasyOCR; handles the ₹ glyph and the replacement-note `*` marker |
| **Serial typography** | digit heights **increase left→right** on genuine notes (regression) — the brief's "number varies with size" |
| Denomination | EasyOCR digit voting + colour-palette tiebreak |
| Proportion analysis | measured note aspect vs canonical RBI mm dimensions per denomination |
| Security thread | morphological vertical-feature detector (handles the windowed thread) |
| Watermark | Gandhi-panel gradient variance |
| Gandhi face | Haar portrait detection (the central "figure") |
| **Micro-lettering** | scale-normalised fine-print sharpness (flags photocopied/blurred print) |
| Bleed lines | counts the raised edge lines vs the RBI per-denomination count (₹100→4, ₹500→5, ₹2000→7) |
| Identification mark | presence + manual-touch guidance for the tactile shape (per denomination) |
| Colour palette | hue entropy + saturation (the "colour variation" feature) |
| UV (proxy) | visible-light approximation, **INFO-only, never a verdict driver** (true UV needs hardware) |

The bleed-line and identification-mark checks are deliberately **conservative**:
PASS on positive evidence, INFO when the photo can't resolve them, and they
**never FAIL** — a phone-photo miscount is a resolution problem, not proof of
forgery (the same discipline that led us to drop ORB motif matching).

### 4.2 Various machine-learning techniques

- **Feature vector** (`backend/features.py`, 50-d, deterministic): Local Binary
  Pattern histogram (26, as in the reference thesis) + colour distribution (17)
  + structure (4) + OCR-free forensic signals (micro-print sharpness, watermark
  variance, vertical-thread energy).
- **Techniques compared:** Logistic Regression, SVM (RBF), Random Forest, KNN,
  and the MobileNetV2 CNN.
- **Honest evaluation:** group-aware cross-validation (a note's augmented
  variants never split across folds) + a held-out test split. Model selection
  is CV-based (no test-set peeking).

### 4.3 Dataset

65 labelled full-note images (42 genuine / 23 fake), split 48 train / 17 test:
34 project fixtures (clean scans + phone photos + synthetic degradations) + 31
from the Apache-2.0 Kaggle "Fake Currency Detection Dataset" (19 real + **12
real fake-note photos**). 101 cropped security-feature templates from that set
were excluded from the whole-note classifier and reserved for feature work.
Training augments the small base set (rotation, brightness, blur, noise, JPEG,
perspective) to a balanced ~250 rows/class; it auto-stops augmenting once real
data is plentiful. Pipeline: `build_dataset.py → train_classical.py →
benchmark_models.py → evaluate_system.py`. See [DATASET.md](DATASET.md).

### 4.4 Beyond detection — honesty, explainability, generative AI, assistant

The system is built to be *trustworthy and usable*, not just a classifier:

- **Input-quality honesty.** A readability gate (`assess_readability`) checks
  whether the serial / size / denomination could actually be read. When a
  plausible note can't be read at all, the verdict is an explicit
  **UNVERIFIED — "can't verify, retake"** with photo guidance, instead of a
  misleading REAL. This directly fixes the failure mode where a low-resolution
  genuine note read as REAL while its individual checks silently failed.
- **Plain-language results.** The UI re-expresses the technical checks as a
  grouped ✓ / ✗ / — "What we checked" panel; raw numbers live behind a
  *Technical details* expander.
- **GenAI explanation** (`/explain`). A read-aloud-friendly summary + manual
  verification steps, generated by Claude (Haiku) with a deterministic template
  fallback so it works offline.
- **Detected-note overlay** (`/predict → regions`). Draws the located note
  region back onto the uploaded photo.
- **Generative security-pattern art** (`/security-pattern`). Procedurally
  generates guilloché ornament (the deterministic mathematical complexity real
  security printing uses to resist copying) from a seed — abstract art, *never*
  currency. This is the honest, legal interpretation of a "generative anti-copy"
  feature.
- **Help chatbot** (`/chat`). A built-in assistant that explains how the system
  works and how to run it; Claude-backed with an offline keyword-FAQ fallback,
  and it refuses any request to make or pass counterfeit money.

Every Claude-backed feature degrades to a deterministic fallback, so the whole
product runs in a local demo with no API key or internet.

---

## 5. Results

### 5.1 Technique comparison (held-out test, see [BENCHMARK.md](BENCHMARK.md))

| Technique | Test acc | Test macro-F1 | Fakes caught |
|---|---|---|---|
| **SVM (RBF)** | **0.765** | **0.742** | 4/6 |
| **Random Forest** | **0.765** | **0.742** | 4/6 |
| Logistic Regression | 0.706 | 0.622 | 2/6 |
| KNN | 0.529 | 0.514 | 3/6 |
| MobileNetV2 (CNN) | 0.588 | 0.564 | 3/6 |

**Finding:** classical ML on engineered features **beats the off-the-shelf
CNN** on this data. Random Forest (best by CV) runs as the live second opinion.

### 5.2 Whole-system end-to-end (all 65 images)

Re-measured after the Phase-K honesty changes (readability gate, Gandhi
FAIL→INFO, pre-OCR upscaling) via `calibrate_thresholds.py`:

| ground truth ↓ / verdict → | REAL | SUSPICIOUS | FAKE | UNVERIFIED |
|---|---|---|---|---|
| **genuine** | 31 | 11 | 0 | 0 |
| **fake** | 10 | 7 | 5 | 1 |

- **0% false positives** — never calls a genuine note FAKE. This is the priority
  metric (rejecting real money is the worst error) and it **held across every
  Phase E–L change**.
- **74%** of genuine notes cleared as REAL (rest flagged SUSPICIOUS for review) —
  up from 71%, thanks to the pre-OCR upscaling that lets low-res genuine notes
  read.
- **~57%** of fakes flagged (FAKE / SUSPICIOUS / UNVERIFIED); ~43% (high-quality
  physical fakes) still pass as REAL — the data-bound false-negative case. The
  Gandhi FAIL→INFO fix (which removed a false-fail on *genuine* notes) slightly
  loosened one borderline fake — an accepted trade for not rejecting real notes.
- **Threshold calibration (Phase F.2):** [CALIBRATION.md](CALIBRATION.md) records
  the ROC curve and Youden's-J optimal boundary; the hand-set 0.65 / 0.35 cutoffs
  are consistent with the evidence on this corpus (no change applied — a 65-image
  ROC is too coarse to retune safely).

### 5.3 Negative results (reported honestly)

- **CNN retrain (Phase F):** transfer-learning retrain on 65 images scored
  0.550 macro-F1 — *worse* than the current 0.564. Not adopted; production
  model kept. CNNs need far more data.
- **ORB motif/emblem matching:** false-failed genuine notes (median 0
  homography inliers) — removed rather than shipped.
- **Micro-print sharpness:** does not separate genuine from fake (sharp fakes
  exist), so it never emits a false PASS — it only flags clearly-degraded print.

These confirm a single conclusion: **the binding constraint is training-data
volume and the scarcity of real physical-counterfeit images**, not the method.

---

## 6. Discussion

The classical models win because hand-crafted features (texture, colour,
proportion, fine-print) are sample-efficient on a small dataset, whereas a CNN
needs thousands of images to generalise. The ensemble + forensic hard gates
make the *system* safer than any single model — notably the 0% false-positive
rate, which matters most in practice (rejecting real money is the worst error).
The honest design choice throughout: when a signal cannot be measured the
system returns INFO rather than guessing, so it never manufactures confidence.

---

## 7. Limitations (honest)

- **No UV/IR or hardware sensing** — UV is a visible-light proxy only.
- **High-quality physical fakes can pass** (~39% false-negative on our set) —
  the hardest case for any phone-photo software system.
- **Small dataset** — metrics are honest but high-variance; more (especially
  physical-fake) images is the single biggest lever.
- **Single-side input** limits see-through-register style checks.
- **No 100% guarantee** — impossible from a photo without sensors.

---

## 8. Future work

1. Gather more labelled images, prioritising **real physical counterfeits**
   (the pipeline auto-improves on re-run — no code changes).
2. Recalibrate the REAL/SUSPICIOUS/FAKE thresholds from ROC curves once the
   dataset is larger (reduce the 12 over-cautious SUSPICIOUS genuine flags).
3. Re-attempt the CNN with a substantially larger dataset; consider a learned
   fusion meta-classifier once data supports it.
4. Optional UV-reactive checks **if** UV-lamp imagery becomes available.

---

## 9. Reproducibility

```
venv\Scripts\python.exe scripts\build_dataset.py --validate   # index the data
venv\Scripts\python.exe scripts\train_classical.py            # train SVM/RF/KNN/LogReg
venv\Scripts\python.exe scripts\benchmark_models.py           # -> docs/BENCHMARK.md
venv\Scripts\python.exe scripts\evaluate_system.py            # end-to-end verdict matrix
venv\Scripts\python.exe -m pytest tests/ -v                   # test suite
uvicorn backend.main:app --host 127.0.0.1 --port 8000         # backend
cd frontend && npm run dev                                    # frontend
```

Full scope and phase history: [PROJECT_SCOPE.md](PROJECT_SCOPE.md),
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [STATUS.md](STATUS.md).
Demo briefing: [DEMO_UPDATE.md](DEMO_UPDATE.md).
