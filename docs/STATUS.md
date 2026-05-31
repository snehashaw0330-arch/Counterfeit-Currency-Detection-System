# Project Status — living handoff

**Read this first in any new session.** It is the single snapshot of where the
project stands. Update it at the **end of every working session** so a fresh
chat resumes with zero context loss. The full plan lives in
[PROJECT_SCOPE.md](PROJECT_SCOPE.md); phase history in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Last updated:** 2026-05-31

---

## Where we are right now

**Phases A, B, C-1 are shipped and green.** The system is a working
FastAPI + Next.js app with a MobileNetV2 classifier and a 10-check forensic
pipeline fused into a REAL/SUSPICIOUS/FAKE verdict.

**Scope just locked** (this session): the forward plan is Phases **D → H** in
[PROJECT_SCOPE.md](PROJECT_SCOPE.md). Two decisions confirmed with the user:
- **ML scope:** compare multiple techniques (CNN vs SVM/RF/KNN/LogReg) with a
  benchmark table + calibrated **rule-based** fusion. Learned meta-classifier
  is a gated stretch goal (needs more data).
- **UV:** stays an honest visible-light proxy (INFO-only). The shared IEEE
  paper and thesis both used visible-light processing only — no UV hardware.
- **Dataset:** public Indian-currency dataset (Kaggle/Mendeley) + our fixtures.

## What works (do not rebuild)
- `/predict` end-to-end: MobileNetV2 + forensic pipeline + combined verdict.
- 10 forensic checks incl. the brief's "number varies with size"
  (`serial_typography_analysis`), proportions, security thread, denomination,
  colour palette, serial OCR (EasyOCR).
- Auto-crop (`_locate_note`), manifest-driven fixtures, ~36 tests, diagnostic
  harness.

## Known gaps (the forward work)
- Only one ML technique → **Phase D**.
- Motifs / micro-lettering / identification-mark / bleed-lines / see-through
  not yet checked → **Phase E**.
- MobileNetV2 provenance opaque, brittle OOD → **Phase F** (retrain).
- README was stale → refreshed this session.

---

## Next concrete action

**Start Phase D.1 — dataset assembly.** Acquire the public Indian-currency
dataset, land it under `dataset/` (git-ignored) with a committed
`dataset/MANIFEST.md`, and write `scripts/build_dataset.py` for a deterministic
70/15/15 stratified split. Acceptance criteria for the whole phase are in
PROJECT_SCOPE.md §4 Phase D.

> Awaiting: go-ahead to begin Phase D implementation (this session produced the
> scope/status docs and refreshed the README; no pipeline code changed yet).

---

## Session log (most recent first)

- **2026-05-31** — Read the full workspace. Authored `docs/PROJECT_SCOPE.md`
  (master scope, phases D–H, brief-traceability matrix, acceptance gates) and
  this STATUS.md. Refreshed stale README. No backend/frontend code changed.

---

## Verification commands (current)
```
venv\Scripts\Activate
python -m pytest tests/ -v          # ~36 tests; Phase-A OCR tests ~2 min
python tests\diagnostic_harness.py  # objective confusion-matrix numbers
uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```
