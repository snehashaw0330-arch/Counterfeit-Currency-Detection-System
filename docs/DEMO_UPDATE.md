# Demo Update — plain-language briefing

For whoever presents the project. Written in normal language so you can explain
it to your instructor/reviewer without needing the code. Full technical detail
is in [PROJECT_SCOPE.md](PROJECT_SCOPE.md) and [STATUS.md](STATUS.md).

---

## The update in 3 points

1. **One clear verdict from many independent checks.** Upload a photo of an
   Indian note and the app says **Real, Suspicious, or Fake** — and it doesn't
   rely on a single black-box model. It combines a deep-learning image model
   with several automatic security checks (serial number, denomination,
   watermark, security thread, note size/proportions, colour, and fine-print
   sharpness), and shows each one to the user so the decision is explainable.

2. **Multiple machine-learning techniques, compared — as the title requires.**
   Alongside the deep-learning model we trained and compared four classical
   models (SVM, Random Forest, KNN, Logistic Regression). On our test set the
   **classical models actually beat the deep-learning model** (best: Random
   Forest), and the app now shows both opinions side-by-side with an
   "agree / disagree" indicator.

3. **Bigger dataset + a repeatable pipeline that improves automatically.** We
   added real genuine and real fake note photos (a public Kaggle dataset) and
   built a one-command pipeline (build → train → benchmark → evaluate). Accuracy
   improves on its own as more note images are added — no code changes needed.

---

## Headline numbers (for questions)

- Technique comparison (held-out test): **Random Forest ≈ 74%** (best), and it
  beats the deep-learning CNN (≈ 56%). Full table: [BENCHMARK.md](BENCHMARK.md).
- Whole-system end-to-end on the full 65-image set:
  - **0 false positives** — never calls a genuine note "Fake".
  - **71% of genuine notes cleared as Real** (the rest flagged "Suspicious" for a
    closer look — cautious, not wrong).
  - **~61% of fakes flagged** (Fake or Suspicious). It misses ~39%, which are
    high-quality fake-note photos that look genuine — the known limit of
    phone-photo software without special hardware or more fake samples.
  - Reproduce anytime: `venv\Scripts\python.exe scripts\evaluate_system.py`.

## If asked "is it perfect?" — the honest answer

No, and we say so deliberately. From a phone photo without special hardware
(UV lamp, etc.) no software can be 100% certain. The system is built to be
**honest**: when a check can't be measured it says "inconclusive" instead of
guessing, and the strongest signal comes from combining many checks plus the
ML models. The main limitation right now is the amount of training data — more
real fake-note images will keep raising the accuracy.

## How to demo (quick)

1. Start backend: `uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. Start frontend: `cd frontend && npm run dev` → open http://localhost:3000
3. Upload a note photo → show the verdict, the per-check panel, and the
   "ML Technique Comparison" panel (CNN vs classical).
