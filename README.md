# Counterfeit Bank Currency Detection — Indian Rupees

Counterfeit Indian banknote detection with **various machine learning
techniques** plus an explainable, visible-light forensic pipeline. A phone
photo of a note returns a single verdict — **REAL / SUSPICIOUS / FAKE** —
backed by per-feature evidence.

> **Planning docs:** the master plan is [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md);
> current state + session handoff is [docs/STATUS.md](docs/STATUS.md);
> phase history is [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md);
> setup is [docs/SETUP_COMMANDS.txt](docs/SETUP_COMMANDS.txt).

---

## Stack

- **MobileNetV2** (Keras) — image classifier (more ML techniques being added; see scope)
- **FastAPI** + uvicorn — `/predict` endpoint ([backend/main.py](backend/main.py))
- **OpenCV** forensic pipeline ([backend/forensic.py](backend/forensic.py))
- **EasyOCR** (primary) + Tesseract (fallback) — serial + denomination OCR
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
- **UV (visible-light proxy)** — honest approximation, INFO-only, never a verdict
  driver (true UV needs hardware — out of scope)

The verdict fuses the ML model and the forensic score
(`0.4·model + 0.6·forensic`) with structural / colour / proportion hard gates.

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
python -m pytest tests/ -v          # ~36 tests (Phase-A OCR tests ~2 min)
python tests\diagnostic_harness.py  # objective confusion-matrix numbers
```

Fixtures are manifest-driven ([tests/sample_notes/manifest.json](tests/sample_notes/manifest.json)).

## Status

Phases A (EasyOCR), B (auto-crop), C-1 (proportions) shipped. Forward work
(multiple ML techniques, motif/micro-lettering/bleed-line checks, model
retraining) is in [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md).
