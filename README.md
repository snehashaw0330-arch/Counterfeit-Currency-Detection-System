# Counterfeit Bank Currency Detection — Indian Rupees

Counterfeit Indian banknote detection with **various machine learning
techniques** plus an explainable, visible-light forensic pipeline. A phone
photo of a note returns a single verdict — **REAL / SUSPICIOUS / FAKE** —
backed by per-feature evidence.

> **Project report:** [docs/REPORT.md](docs/REPORT.md) (submission-grade capstone).
> **Planning docs:** the master plan is [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md);
> current state + session handoff is [docs/STATUS.md](docs/STATUS.md);
> phase history is [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md);
> setup is [docs/SETUP_COMMANDS.txt](docs/SETUP_COMMANDS.txt).

---

## Stack

- **MobileNetV2** (Keras) — deep image classifier
- **Classical ML second opinion** — SVM / Random Forest / KNN / Logistic
  Regression on engineered features, benchmarked in [docs/BENCHMARK.md](docs/BENCHMARK.md)
- **FastAPI** + uvicorn — `/predict`, `/diagnose`, `/explain`, `/chat`,
  `/security-pattern` ([backend/main.py](backend/main.py))
- **OpenCV** forensic pipeline ([backend/forensic.py](backend/forensic.py))
- **EasyOCR** (primary) + Tesseract (fallback) — serial + denomination OCR
- **Claude (Anthropic)** — plain-language explanations + help chatbot, each with
  a deterministic offline fallback so the demo works with no API key
- **Next.js** + Tailwind — frontend ([frontend/app/page.tsx](frontend/app/page.tsx))

## Forensic checks (implemented, not placeholders)

Each returns `PASS` / `FAIL` / `INFO` with the measured numbers shown:

- **Structural sanity** — hard gate (aspect, edges, brightness, dark quadrants)
- **Serial number (OCR)** — EasyOCR, handles the ₹ glyph and replacement-note `*`
- **Serial typography** — digit sizes ascend left→right on real notes (the brief's
  "note number varies with size" feature)
- **Denomination** — EasyOCR digit voting + colour-palette tiebreak
- **Proportion analysis** — measured note aspect vs canonical RBI mm dimensions
- **Security thread** — morphological detector (handles the windowed thread)
- **Watermark** — Gandhi-panel gradient variance
- **Gandhi face** — Haar portrait detection
- **Colour palette integrity** — hue entropy + saturation (colour-variation feature)
- **Micro-lettering / fine print** — high-frequency texture energy (honest:
  FAIL on lost detail, INFO when resolution is too low, never a false PASS)
- **Bleed lines** — counts the raised edge lines vs the RBI per-denomination
  count (₹100→4, ₹200→4, ₹500→5, ₹2000→7)
- **Identification mark** — presence + manual-touch guidance for the tactile
  shape (triangle/circle/… per denomination)
- **UV (visible-light proxy)** — honest approximation, INFO-only, never a verdict
  driver (true UV needs hardware — out of scope)

The verdict fuses the ML model and the forensic score
(`0.4·model + 0.6·forensic`) with structural / colour / proportion hard gates.
When the photo is too unclear to actually read the note (no serial, no size, no
denomination), the system returns an honest **UNVERIFIED — "can't verify,
retake"** instead of a misleading REAL.

## Beyond detection (demo features)

- **Plain-language results** — a "What we checked" panel rewrites the technical
  checks into ✓ / ✗ / — with human descriptions; raw numbers live behind a
  *Technical details* expander.
- **Explain with AI** (`/explain`) — a read-aloud-friendly summary + manual
  verification steps (Claude, with a deterministic template fallback).
- **Detected-note overlay** — draws the located note region on your upload.
- **Security Pattern Studio** (`/security-pattern`) — procedurally generates
  guilloché (anti-copy) art from a seed; abstract art, not currency.
- **Help chatbot** (`/chat`) — a floating assistant that answers "how does it
  work / how do I run it / what does this verdict mean".
- **Listen + Hindi** — a "Listen" button reads the result aloud (Web Speech) and
  an English/हिंदी toggle localises the verdict and explanation (accessibility).
- **Downloadable PDF report** — a print-ready one-page verdict report.
- **Live camera capture** — analyse a note straight from the webcam/phone camera.
- **AI heatmap (Grad-CAM)** — overlay showing where the CNN looked.
- **Digital-tamper check (ELA)** — flags possible digitally edited/spliced images
  (informational; distinct from physical counterfeits).

## Quick start

```bash
# Backend
venv\Scripts\Activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000   # http://127.0.0.1:8000/docs

# Frontend
cd frontend && npm install && npm run dev               # http://localhost:3000
```

First backend launch downloads ~64 MB of EasyOCR weights (one-time). Full
setup, prerequisites, and troubleshooting: [docs/SETUP_COMMANDS.txt](docs/SETUP_COMMANDS.txt).

## Tests

```bash
python -m pytest tests/ -v          # full unit + API + phase suites (Phase-A OCR ~2 min)
python scripts\evaluate_system.py   # end-to-end verdict confusion matrix
```

Fixtures are manifest-driven ([tests/sample_notes/manifest.json](tests/sample_notes/manifest.json)).

## Status

Shipped: A (EasyOCR), B (auto-crop), C-1 (proportions), **D** (multiple ML
techniques + benchmark), **E** (micro-print, bleed lines, identification mark),
**F** (CNN retrain — evaluated, kept the stronger model), **G** (diagnose,
validation, launcher, detected-note overlay), **H** (capstone report), **I**
(GenAI explanation), **J** (security-pattern art), **K** (honest UNVERIFIED
verdict + plain-language UI), **L** (help chatbot).

Honest limitations (see [docs/REPORT.md](docs/REPORT.md) §limitations): accuracy
is bound by a small (~65-image) dataset — ~0% false positives but some high-
quality fakes pass; UV is a visible-light proxy; single-side, Indian Rupees
only. The biggest improvement lever is more data, especially real physical
counterfeits. Live state + handoff: [docs/STATUS.md](docs/STATUS.md).
