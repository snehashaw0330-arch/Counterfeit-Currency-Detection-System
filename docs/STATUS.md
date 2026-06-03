# Project Status — living handoff

**Read this first in any new session.** It is the single snapshot of where the
project stands. Update it at the **end of every working session** so a fresh
chat resumes with zero context loss. The full plan lives in
[PROJECT_SCOPE.md](PROJECT_SCOPE.md); phase history in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Last updated:** 2026-06-03 (Phase L — help chatbot; full-roadmap build in progress)

## 🚧 Full-roadmap build (user mandate: "complete everything, no loose ends, demo-perfect")

Working in order, one solid commit per phase:
- **Phase L (help chatbot) — DONE.** `backend/chatbot.py` (`answer()` → {reply,
  source}; Claude Haiku w/ cached project-knowledge system prompt when key set,
  deterministic keyword-FAQ fallback otherwise; safety refusal for "make fake
  money"; never raises). `POST /chat` in main.py. Floating `ChatAssistant` widget
  in the frontend (suggestions, history, source-aware). 18 FAQ tests green;
  `tests/test_phase_l_chat.py`; tsc + next build clean.
- **Next:** Phase E.3 identification mark, E.4 bleed lines, E.1 motif (INFO),
  E.5 see-through; Phase G.3 region overlays; Phase F.2 threshold calibration;
  demo polish (gallery, README/REPORT). All honest (INFO when not assessable;
  never false-FAIL a genuine note).



## ✅ Phase K — readability gate, verdict honesty, plain-language UI

**Trigger:** user tested a genuine ₹100 web-thumbnail photo and got OCR "Not
Detected", proportion "couldn't measure", Gandhi face FAIL — yet verdict REAL
82%, plus a wall of technical INFO/jargon. Root-caused (verified): the note
wasn't auto-located and/or was too low-res, so OCR/face/proportion collapsed,
but the *other* checks still passed → misleading REAL. Plus the output was
engineer-speak.

**Fixes (all shipped):**
- **K2 read harder:** `_easyocr_words` now upscales small crops to
  `_TARGET_OCR_WIDTH` BEFORE EasyOCR (was only on the dead Tesseract path) —
  this alone made the ₹100 serial read ("7MP 979885"). `_detect_note_quad`
  gained a `minAreaRect` fallback (rounded/rotated/soft edges). Gandhi "no face"
  downgraded **FAIL → INFO** (was false-failing genuine notes; same rule as
  micro-print/ORB).
- **K1+K3 honesty gate:** `forensic.assess_readability(image, results)` →
  level full/partial/none from whether serial/proportions/denomination read.
  `main._analyze` overlays it: a REAL/SUSPICIOUS that read NOTHING becomes
  **`UNVERIFIED`** ("Can't verify — retake"); a clear FAKE (structural/colour)
  stays FAKE. New response fields: `security_verdict`, `verification_level`,
  `verification{...}`, `guidance`.
- **K5 plain UI** (`frontend/app/page.tsx`): verdict **banner** (plain headline +
  ₹denom + retake tips for UNVERIFIED + limited-check caveat for partial),
  **"What we checked"** grouped plain findings (✓/✗/— + human text, 4 groups),
  AI explanation promoted, and ALL raw cards moved into a **"Technical details"**
  `<details>` expander. `tsc` + `next build` both clean.
- **K6 tests:** `tests/test_phase_k_quality.py` (level logic, unread/guidance,
  never-raises, blank→none, endpoint exposes verification). Updated `test_api.py`
  for the UNVERIFIED enum + verification fields. `scripts/evaluate_system.py`
  updated to count UNVERIFIED as a 4th outcome. **31 passed** (phase_k + api +
  forensic).

**Verified end-to-end on the user's ₹100:** REAL 93% / verification **full** /
serial **7MP 979885** / proportions PASS / face PASS / guidance empty. All three
reported defects fixed.

## ✅ Phases I + J — DONE, shipped end-to-end

**Phase I (GenAI explanation) — committed (b70f89d):**
- `backend/genai.py` — `explain(result)` → {summary, reasons, manual_checks, source}.
  Uses Claude **Haiku** (`claude-haiku-4-5`) with a cached system prompt +
  JSON-schema output when `ANTHROPIC_API_KEY` is set; **deterministic template
  fallback** otherwise. Never raises. (Built via the claude-api skill.)
- `backend/main.py` — `POST /explain` (takes a /predict result, returns the
  explanation; never fails the request) + the genai imports.
- `requirements.txt` — added `anthropic` (installed: 0.105.2).
- `tests/test_phase_i_genai.py` — **13 tests green**.
- `frontend/app/page.tsx` — **`ExplainPanel`** ("Explain with AI" button → POST
  /explain; shows summary + reasons + manual checks; AI vs RULE-BASED source badge).
- This is the project's honest "GenAI that does something good": explainability +
  accessibility (read-aloud), NOT counterfeit generation.

**Phase J (generative security-pattern art) — shipped:**
- `backend/security_pattern.py` — procedural **guilloché / rosette / micro-text**
  generator. Spirograph (hypotrochoid) rosettes + sinusoidally-modulated woven
  rings + a seed-derived micro-text ring, via numpy + PIL. `generate_pattern(seed,
  size)` / `pattern_png(...)`. Seed accepts int OR string (SHA-256 → stable int,
  so cross-machine reproducible). Size clamped [128,1200]. **Never raises**
  (deterministic `_fallback_pattern`). **Abstract ornament only — NOT currency.**
- `backend/main.py` — `GET /security-pattern?seed=...&size=...` → PNG (`Response`,
  `Query` validation; out-of-range size → 422, never 500).
- `tests/test_phase_j_pattern.py` — **15 tests green** (PNG magic bytes, determinism,
  varies per seed, int==numeric-string, size clamp, never-raises on 7 weird seeds,
  endpoint determinism/variation, 422 on bad size).
- `frontend/app/page.tsx` — **`SecurityPatternStudio`** card (seed input → live
  `<img>` of the generated guilloché). Sample: `docs/sample_security_pattern.png`.
- Rationale: user kept asking for "GenAI that makes notes that can't be cloned."
  That's not legally/literally buildable; this is the legitimate version — the
  deterministic mathematical complexity (guilloché) that real security printing
  uses, generated as abstract art. **Does NOT generate currency imagery.**

## Phase F result + capstone report (latest)

- **Phase F (retrain CNN) — DONE, negative.** `scripts/train_cnn.py`: transfer-
  learning retrain (frozen MobileNetV2 + head, augmented) scored **0.550 test
  macro-F1 vs current 0.564** — worse (catches 1/6 fakes). **Not adopted**;
  production model kept. `models/*_v2.keras` git-ignored. Third confirmation of
  the data ceiling (after weak Phase-E checks + failed ORB motifs).
- **Capstone: `docs/REPORT.md`** — submission-grade project report (abstract,
  architecture, methodology, brief-traceability, full results incl. the honest
  negative results, limitations, future work, reproducibility).
- **Verdict recalibration (Phase F.2) is the main remaining model-side task** —
  deferred until the dataset is larger (ROC-based threshold selection).

## Phase E so far (visible-light security features)

- **Shipped:** `analyze_microprint` (fine-print / micro-lettering) — evidence-honest
  (FAIL on lost detail, INFO otherwise, **never a false PASS**; calibration showed
  sharp fakes score as high as genuine). Wired into pipeline + `EXPECTED_KEYS` +
  frontend type/card. New key `microprint_detection`. Tests in
  `tests/test_phase_e_microprint.py`.
- **Folded forensic signals into the ML vector** (user choice): `backend/features.py`
  now 50-dim (+microprint sharpness, +watermark variance, +vertical-thread energy,
  all log1p). Refactored `_watermark_variance` out of `detect_watermark` to share.
  Retrained → **Random Forest improved to test macro-F1 0.742** and is the live
  second opinion. Benchmark refreshed.
- **Tried and REJECTED (honestly):** ORB emblem/motif matching — calibration showed
  it false-fails genuine notes (median 0 inliers), so removed rather than shipped.
  Lesson recorded: single hand-crafted visible-light checks are weak/unreliable on
  this corpus; discrimination lives in the ML ensemble; **data is the binding
  constraint**.
- **New:** `scripts/evaluate_system.py` — end-to-end /predict verdict eval over the
  corpus (the real product metric). `docs/DEMO_UPDATE.md` — plain-language 3-point
  briefing for the demo presenter.

**End-to-end result (65 imgs, full /predict verdict):** genuine → 30 REAL / 12
SUSPICIOUS / 0 FAKE; fake → 9 REAL / 8 SUSPICIOUS / 6 FAKE. Headline: **0%
false positives**, 71% genuine cleared, **61% of fakes flagged**, 39% of fakes
(high-quality physical fakes) pass as REAL — the data/hardware ceiling.

## Latest data + benchmark (Kaggle "Fake Currency Detection Dataset")

Corpus is now **65 images (42 genuine / 23 fake)**, 48 train / 17 test — up from
34. The Kaggle set added **19 real + 12 fake FULL notes** (the 12 fakes are real
fake-note *photos*, not synthetic). Its **101 feature-crop images** (`*_Features*`)
were auto-excluded from the whole-note classifier and **kept under
`dataset/Dataset/` for Phase E** (they are the id1–id6 security-feature templates,
matching the thesis approach). Layout: full notes moved to
`dataset/real/kaggle_500|2000` and `dataset/fake/kaggle_500|2000`.

**Refreshed benchmark (held-out 17-image test, see docs/BENCHMARK.md):**
| Technique | Test acc | Test macro-F1 | Fakes caught |
|---|---|---|---|
| **SVM (RBF)** | **0.765** | **0.742** | **4/6** |
| Random Forest | 0.706 | 0.689 | 4/6 |
| Logistic Regression | 0.706 | 0.622 | 2/6 |
| KNN | 0.647 | 0.614 | 3/6 |
| MobileNetV2 (CNN) | 0.588 | 0.564 | 3/6 |

Headline: **classical ML on engineered features beats the off-the-shelf CNN**;
SVM best on test. Model selection stays CV-based (no test peeking) — best by CV
is recorded in metrics.json and used as the live second opinion.

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

## Phase D — DONE (D.1–D.6), end to end

The "Various ML Techniques" requirement is now real and live:
- **D.1** `scripts/build_dataset.py` → `dataset/index.json` (34 imgs: 23 genuine /
  11 fake, 25 train / 9 test, deterministic, auto-ingests `dataset/real|fake/`).
- **D.2** `backend/features.py` — 47-dim feature vector (LBP-26 + hue-12 +
  colour-5 + structure-4). 6 tests.
- **D.2.5** `backend/augment.py` — seeded augmentation. 5 tests.
- **D.3** `scripts/train_classical.py` — SVM/RF/KNN/LogReg, **group-aware** CV
  (StratifiedGroupKFold — fixed an augmentation-leakage flaw that had inflated CV
  to 0.99). Saves `models/classical/*.joblib` (git-ignored) + `metrics.json`.
- **D.4** `scripts/benchmark_models.py` → `docs/BENCHMARK.md` (committed).
- **D.5** `backend/classical.py` + `main.py` — best classical model runs as a
  live **second opinion** in `/predict` (`ml_models` block). Verdict logic
  UNCHANGED (fusion-weight recalibration deferred to Phase F). 3 tests.
- **D.6** frontend `ModelComparisonPanel` — CNN vs classical + agreement badge.

**Benchmark headline (held-out 9-image test):** Random Forest best (test acc
0.778, macro-F1 0.750 — only model that catches fakes). **MobileNetV2 CNN
collapses to "all genuine" (catches 0/3 fakes, macro-F1 0.400)** — concrete
evidence for why multiple techniques + forensics are needed. Numbers are modest
and honestly capped by the ~34-image corpus; CV is now honest (RF 0.668).

Also fixed a **pre-existing test bug**: `test_forensic.py` EXPECTED_KEYS was
missing `serial_typography_analysis` (shipped in 45a7ec7 but never added to the
guard). Now 15/15.

## Next concrete action

Dataset ingested + benchmark refreshed (see above). Pick the next phase:
- **Phase E** — visible-light security features. **Now has ready-made templates:**
  the Kaggle `*_Features` crops under `dataset/Dataset/` are exactly the id1–id6
  security regions — use them as match references for motif / micro-lettering /
  identification-mark checks. No further data needed to start.
- **Phase F** — retrain MobileNetV2 on the 65-image set + recalibrate the fused
  verdict (the CNN is now the weakest technique — 0.564 — so this has clear upside).

**Still the biggest lever: more data**, especially real *physical* counterfeits
(the 12 Kaggle fakes help; ingestion auto-scales — just drop more in and re-run
the 3 scripts). Guide: [DATASET.md](DATASET.md).

Then choose the next phase:
- **Phase E** — visible-light security features (motifs, micro-lettering,
  identification mark, bleed lines, see-through). Scope §4.
- **Phase F** — retrain MobileNetV2 on the dataset + recalibrate the fused
  verdict (high value given the CNN's poor showing above).

> **Install when Phase F / plotting starts:** pandas, matplotlib, seaborn
> (declared in requirements; BENCHMARK.md currently uses plain-markdown tables,
> so not needed yet).

## Open thread — generative AI?

User asked whether the shared docs require a "GenAI that creates notes that
can't be counterfeited." **They do not** — both the IEEE paper and the thesis
are detection-only; no generation, no GenAI. Discussed the legitimate adjacent
use: generative *data augmentation* (synthetic tampered/fake samples) to harden
the detector — optional, defensible, would also count as a "various ML
technique." NOT building a realistic-counterfeit generator (illegal/misusable).
Decision pending from user on whether to add an optional augmentation phase.

---

## Session log (most recent first)

- **2026-06-03 (10)** — **Phase K: trustworthy + understandable results.** User
  reported a genuine ₹100 photo reading as REAL while OCR/proportion/face all
  silently failed, with jargon-heavy output. Root-caused to localization +
  low-res OCR; fixed by (a) upscaling before EasyOCR + a minAreaRect localization
  fallback + Gandhi FAIL→INFO, (b) an honest readability gate that yields a new
  `UNVERIFIED` "can't verify — retake" verdict instead of a misleading REAL, and
  (c) a plain-language UI redesign (verdict banner + grouped "what we checked"
  findings + retake tips + technical-details expander). 31 tests green; frontend
  `next build` clean; verified on the user's actual ₹100 (now REAL/full, serial
  "7MP 979885"). Eval script updated for the 4th verdict.
- **2026-06-02 (9)** — **Phases I + J shipped end-to-end.** Phase I: tested
  (`tests/test_phase_i_genai.py`, 13 green) and committed the GenAI explanation
  layer (`backend/genai.py` + `POST /explain`) and added the frontend
  `ExplainPanel`. Phase J: built `backend/security_pattern.py` (procedural
  guilloché/rosette/micro-text art, deterministic per seed, never raises),
  `GET /security-pattern` PNG endpoint, `tests/test_phase_j_pattern.py` (15
  green), and the frontend `SecurityPatternStudio` card. Frontend typechecks
  clean. 49/49 in the I+J+api+forensic subset green. Both are the honest
  answer to the user's "GenAI" ask — explainability + abstract security-pattern
  art, never counterfeit generation.
- **2026-06-01 (8)** — Phase G (backend half): refactored `/predict` verdict
  into a shared `_analyze()`; added upload validation (empty / 25 MB cap /
  unreadable → clean error, no 500s); new **`/diagnose`** endpoint (superset of
  /predict + raw EasyOCR tokens via `forensic.diagnostics()`). 6/6 API tests
  green (incl. 2 new). Remaining Phase G: frontend region overlays (needs the
  backend to return bbox coords), one-command launch script.
- **2026-06-01 (7)** — Phase F: retrained MobileNetV2 (transfer learning) →
  0.550, worse than 0.564, NOT adopted (data ceiling). Wrote capstone
  `docs/REPORT.md`. Git-ignored `models/*_v2.keras`.
- **2026-06-01 (6)** — Phase E: shipped honest micro-print check; folded 3 OCR-free
  forensic signals into the 50-dim ML vector (RF → 0.742, now live 2nd opinion);
  rejected unreliable ORB motif matching (would false-fail reals). Added
  `scripts/evaluate_system.py` + `docs/DEMO_UPDATE.md`. 33+ tests green.
- **2026-05-31 (5)** — Ingested Kaggle "Fake Currency Detection Dataset". Added
  exclusion of feature-crop/template/checkpoint folders to build_dataset.py;
  reorganised full notes into dataset/real|fake (kept feature crops under
  dataset/Dataset/ for Phase E). Corpus 34→65 imgs. Retrained + re-benchmarked:
  SVM best (test macro-F1 0.742); all classical models beat the CNN (0.564).
  Fixed stale hardcoded caveat in train_classical.py (now reports real counts).
- **2026-05-31 (4)** — Data-prep hardening (user chose "more data first").
  Rewrote `scripts/build_dataset.py` to recursively ingest any folder layout
  under `dataset/` with path-based label inference + `--validate` corrupt-image
  pass; confirmed no regression on fixtures (still 34/25/9). Wrote
  `docs/DATASET.md` (drop-in rules + suggested Kaggle/Mendeley datasets).
- **2026-05-31 (3)** — **Phase D shipped end-to-end (D.1–D.6).** Dataset indexer,
  feature extractor, augmentation, classical training (group-aware CV),
  benchmark report, live second-opinion in `/predict`, frontend comparison
  panel. Generated `docs/BENCHMARK.md`. Fixed pre-existing EXPECTED_KEYS test
  bug. 33 tests verified green this session (forensic 15, D-features 6,
  D-augment 5, D-classical 3, api 4). Confirmed: dataset/ did not exist; the
  corpus is the tests/sample_notes fixtures (~34 imgs). Answered user: GenAI to
  "create uncounterfeitable notes" is NOT in the shared docs; offered honest
  generative-augmentation instead (now used in D.3 training).
- **2026-05-31 (2)** — **Phase D.2 shipped.** Installed scikit-learn + joblib
  (venv was out of sync with requirements). Built `backend/features.py` —
  deterministic 47-dim hand-crafted feature vector (LBP-26 + hue-hist-12 +
  colour-scalars-5 + structure-4), never-raises contract. Added
  `tests/test_phase_d_features.py` (6 tests, all green in ~5 s). Added
  scikit-image + joblib to requirements.txt.
- **2026-05-31 (1)** — Read the full workspace. Authored `docs/PROJECT_SCOPE.md`
  (master scope, phases D–H, brief-traceability matrix, acceptance gates) and
  this STATUS.md. Refreshed stale README. No backend/frontend code changed.

---

## Verification commands (current)
```
venv\Scripts\Activate
python -m pytest tests/ -v          # ~44 tests; Phase-A OCR tests ~2 min
python tests\diagnostic_harness.py  # objective confusion-matrix numbers

# Phase D pipeline (re-run in this order after adding dataset images):
python scripts\build_dataset.py     # -> dataset/index.json
python scripts\train_classical.py   # -> models/classical/*.joblib + metrics.json
python scripts\benchmark_models.py  # -> docs/BENCHMARK.md

uvicorn backend.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

> Not re-run this session (unaffected by Phase D — no OCR/locate/proportion code
> changed): test_phase_a_ocr (~2 min), test_phase_b_locate, test_phase_c_proportions.
> The full diagnostic harness wasn't re-run because the verdict logic is unchanged
> (classical model is display-only); test_api's blank→not-REAL guard still passes.
