# Project Status — living handoff

**Read this first in any new session.** It is the single snapshot of where the
project stands. Update it at the **end of every working session** so a fresh
chat resumes with zero context loss. The full plan lives in
[PROJECT_SCOPE.md](PROJECT_SCOPE.md); phase history in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Last updated:** 2026-06-28 (big requirement kicked off: Phase R self-auth
note scheme + Phase S multi-currency/polymer; two-agent parallel split)

## 🆕 Active effort — Phase R + Phase S (two agents in parallel)

> **Working checklist + scorecard: [PHASE_RS_PROGRESS.md](PHASE_RS_PROGRESS.md)** —
> what's done, what's left (T1–T5), addressed one item at a time.

The big new requirement (serial-guilloché generation + verification, digital
PUF, foreign/polymer multi-currency detection) is split into two contract
phases — see [PROJECT_SCOPE.md](PROJECT_SCOPE.md) §4 (Phase R, Phase S, and the
two-agent file-ownership table). Decisions locked with the user:
- **Guilloché/PUF = closed-loop PoC** (a proposed self-authenticating note),
  NOT verification of real RBI notes.
- **Multi-currency = formally in scope** (Phase S); reverses the old INR-only
  non-goal. Counterfeit benchmarking honestly limited to **INR + BDT** (no
  public polymer-counterfeit data).
- **Two-agent split:** Agent C owns foreign/data NEW modules (`backend/country.py`,
  `backend/polymer.py`, BankNote-Net scripts, `dataset/foreign/**`, foreign
  docs). Agent A (me) owns Phase R + ALL integration/contract files
  (`main.py`, `forensic.py`, `frontend`, `EXPECTED_KEYS`, scope/status/report/
  dataset docs) and wires C's modules into `/predict` behind a country gate.

**Data landed so far:** JaalTaka (Bangladesh, real + physical counterfeit) via
`scripts/fetch_jaaltaka.py` — partial-zip HTTP-range fetch, validation slice of
96 imgs verified (0 corrupt). BankNote-Net embeddings + a 17-currency classifier
(95% in-distribution) fetched/trained by Agent C — **embeddings-only, genuine-
only, no encoder shipped, no BDT class** → kept as an OFFLINE benchmark, not the
live country-detection path. Dataset layout locked to
`dataset/foreign/<ccy>/{full_note,security_crops}/{real,fake}` + metadata.

**Agent A progress (all additive; INR `/predict` verdict path contract-stable,
test_api green, no regression):**
- **R.1 DONE** — serial→guilloché secure-note token (`backend/security_pattern.py`
  `secure_note_token/png`, crisp OCR-legible header) + `GET /secure-note/generate`.
- **R.2 DONE** — closed-loop guilloché verification (`backend/secure_note.py`,
  native-scale SSIM, calibrated 3-band AUTHENTIC/TAMPERED/UNVERIFIED, OCR-trust
  rule so an OCR'd serial never yields a false TAMPERED) + `POST /secure-note/verify`.
- **R.3 DONE** — digital PUF (`backend/puf.py`, texture-fingerprint Hamming hash,
  calibrated 0.25 threshold: same-capture ≤0.04, different ≥0.36; local JSON
  registry, git-ignored) + `POST /puf/enroll` + `POST /puf/verify`.
- **S.2/S.3 integration DONE** — `backend/foreign_routing.py` wires Codex's
  `country.py`+`polymer.py` into `/predict` behind a gate (detector runs ONLY
  when the note isn't identified as INR → zero cost on the common path; confident
  foreign → country+polymer fields + verdict forced UNVERIFIED). `POST /detect-country`.
- **Tests:** test_phase_r_secure_note(9), _verify(13), _verify_ocr(3 slow),
  _puf(12), test_phase_s_integration(6), + Codex country/polymer(13) + api(6) — green.

**Next build steps (Agent A):** R.4 frontend (SecureNoteStudio generate+verify,
PUF enroll/verify, country/polymer display) → R.5 docs (GENAI_EXPLAINED/REPORT,
PROJECT_SCOPE acceptance) → normalize BDT tree to locked layout
(`dataset/foreign/bdt/security_crops/{real,fake}`) + conform `fetch_jaaltaka.py`
+ pull full slice (validate_foreign_dataset.py flags the current old layout).



## ⏭️ NEXT SESSION — resume here

**Author:** the project is built by **Sneha Shaw** (see [[project_author]] memory).
The chatbot credits her and has an easter egg: "hey sneha this side" → "Hello
boss!"; "who is sneha / her gender / what does she like" → "a girl who loves ice
cream 🍦 and sleeping 😴". Wired in `backend/chatbot.py` (offline intents +
`_PROJECT_KNOWLEDGE`). **Backend restart needed** for /chat changes to show.

**Sneha picked the big-upgrade roadmap = Track A (Accuracy & Robustness):**
robust note localization (item 1) + dataset expansion & CNN retrain (item 2).

**DONE this session (uncommitted — one commit each pending user OK):**
1. **Chatbot voice INPUT — DONE.** Mic button in `ChatAssistant`'s input row
   (`frontend/app/page.tsx`) using Web Speech API
   (`SpeechRecognition`/`webkitSpeechRecognition`). Feature-detected client-side
   (hidden on Firefox), default lang **en-IN**, `interimResults` fill the box
   live + append to typed text, red pulsing "Listening…" state, mic released on
   send/unmount. Minimal local TS typings (no `any`). tsc + next build clean.
2. **Track A item 1 — Robust note localization — DONE.** `_detect_note_quad` is
   now a 2-stage orchestrator in `backend/forensic.py`: (a) `_detect_note_quad_edges`
   (the original Canny/contour + rotated-rect, byte-for-byte unchanged), then
   (b) `_segment_note_quad` — GrabCut foreground segmentation seeded by
   `_spectral_residual_saliency` (FFT, no opencv-contrib) + a central prior,
   used ONLY when the edge path returns None. Strictly additive. Result memoized
   by a 24×24-thumbnail key (`_QUAD_CACHE`, LRU≤6) so the ~5 detect calls/predict
   don't each re-run GrabCut. All consumers (`_locate_note`, `note_region`,
   `analyze_proportions`, `assess_readability`) benefit for free.
   **Validated:** `tests/test_phase_q_localization.py` (7 tests) green; localization-
   only pass over 36 real samples → **0 regressions** (all 14 edge-located still
   located) + **5 new correct recoveries** (handheld phone ₹500, phone ₹50,
   specimen ₹50, ₹20, a multi-note ₹100 composite — visually confirmed each crop
   is the actual note). Garbage-safe (None on blank/white/noise).

**PENDING — Track A item 2 (Dataset expansion + CNN retrain). BLOCKED on user:**
- **(a) Data fetch (the real accuracy lever):** `dataset/` is git-ignored and
  currently only **65 labelled imgs (42 genuine / 23 fake, mostly ₹500/₹2000)**.
  Retraining on 65 won't move the ceiling — it's data-bound. Sneha must download
  the Kaggle/Mendeley sets (login + ToS; an agent can't) per `docs/DATASET.md`
  §3, drop them under `dataset/real|fake/`, then re-run `build_dataset.py`.
- **(b) Heavy-compute window:** retrain/benchmark/calibrate are CPU-heavy passes
  — do NOT run while the local app is open (backend-starvation lesson,
  [[no-heavy-bg-jobs-during-use]]). Run when the app is idle.
- Pipeline is data-driven & already built: `scripts/build_dataset.py --validate`
  → `scripts/train_cnn.py` (MobileNetV2 head → `..._v2.keras`, only adopt if test
  macro-F1 beats current 0.564) → `scripts/benchmark_models.py` →
  `scripts/calibrate_thresholds.py`. No code changes needed to ingest more data.

**Other strong roadmap candidates (not picked this round):** region feature
overlays, batch mode + CSV export, session history gallery, serial-duplicate
detection, more languages, PWA/installable, full voice conversation, Dockerize.

**Camera misread root cause (for the localization work):** a note that's small/
tilted in a busy, low-light frame isn't isolated by the contour-based
`_detect_note_quad`, so OCR reads stray digits. Phase-O now crops to the green
guide box, but true robustness needs real segmentation (item 2a above).

## ✨ New features M–P (user picked all four; built in order, one commit each)

## ✨ New features M–P (user picked all four; built in order, one commit each)

- **M — Accessibility: Listen + Hindi (4383b24).** Language toggle (EN/हिंदी);
  `genai.explain(result, lang)` full Hindi template + Hindi Claude path; Hindi
  verdict headlines; Web-Speech "Listen" buttons (verdict banner + explanation)
  reading aloud in the chosen language. Tests +4.
- **N — Downloadable PDF report (9767293).** "Download report (PDF)" opens a
  self-contained print-ready doc (verdict, denom, confidence, serial, the note
  image, full findings table, disclaimer) → browser Save-as-PDF. No new deps.
- **O — Live camera capture (11bb3c2).** "Use camera" → `CameraModal`
  (getUserMedia rear cam, framing guide) captures a frame → same /predict flow.
- **P.1 — Digital-tamper / ELA (1c8a2dc).** `analyze_tamper_ela` on the ORIGINAL
  image; INFO-only (heuristic, never drives the verdict). Pipeline now 15 checks.
- **P.2 — Grad-CAM heatmap (285f3d2).** `backend/gradcam.py` (best-effort, finds
  Conv_1 7×7×1280, None on failure); `/predict` returns a `heatmap` RGBA PNG data
  URL; frontend "Show AI heatmap" overlay. Verified end-to-end on a real ₹100.

All: tsc + next build clean; new phase tests green.

## 🚧 Full-roadmap build (user mandate: "complete everything, no loose ends, demo-perfect")

Working in order, one solid commit per phase:
- **Phase L (help chatbot) — DONE (72dd33b).** `backend/chatbot.py` + `POST /chat`
  + floating `ChatAssistant` widget. Claude Haiku w/ cached project-knowledge
  prompt when key set; deterministic keyword-FAQ fallback; safety refusal; never
  raises. 18 tests.
- **Phase E (tactile security features) — DONE.** `analyze_bleed_lines` (counts
  edge bleed lines vs the RBI per-denomination count: 100→4, 200→4, 500→5,
  2000→7) and `analyze_identification_mark` (presence + manual-touch guidance,
  shape per denom). Both **honest**: PASS on positive evidence, INFO otherwise,
  **never FAIL** (a phone-photo miscount is a resolution problem, not forgery).
  Wired into pipeline + EXPECTED_KEYS (now 14 checks) + frontend type/grid/plain
  findings. `tests/test_phase_e_tactile.py` (15). NOTE: E.1 motif and E.5 see-
  through were deliberately NOT shipped as hollow always-INFO checks (no
  placeholders) — figures/motifs are covered by the portrait check; see-through
  stays a documented single-side limitation (scope §5).
- **Phase G.3 (detected-note overlay) — DONE.** `forensic.note_region()` →
  normalised [0,1] note polygon in ORIGINAL coords; `/predict` returns `regions`;
  frontend draws a green SVG overlay + "Detected note" tag on the upload.
  `tests/test_phase_g_regions.py` (4).
- **UI makeover — DONE (bf28627).** Full visual redesign: glass design system
  (globals.css ambient glows + Geist + .card), gradient hero, drag-and-drop
  dropzone, gradient Detect button w/ spinner, translucent panels, SVG chat FAB
  + polished chat panel. tsc + next build clean. (User: previous UI "looked like
  a kid's UI" — this replaces it.)
- **Phase F.2 (threshold calibration) — DONE.** `scripts/calibrate_thresholds.py`
  runs /predict over the corpus, prints the verdict confusion matrix, computes
  ROC-AUC + Youden's-J optimal boundary, writes `docs/CALIBRATION.md`. Finding:
  hand-set 0.65/0.35 cutoffs are consistent with the evidence (no change; 65-img
  ROC too coarse to retune). Encoding fix: `sys.stdout.reconfigure(utf-8)` for
  the → / ₹ glyphs on Windows.
- **No-regression re-measure (post Phase E–L):** genuine 31 REAL / 11 SUSP / 0
  FAKE / 0 UNVER (**0% false positives held**, clearing 71%→74%); fake 10 REAL /
  7 SUSP / 5 FAKE / 1 UNVER (fakes-as-REAL 9→10, the accepted Gandhi-INFO trade).
- **Full suite GREEN: 137 passed** (~9 min, incl. the live Phase-A OCR tests
  that exercise the changed OCR path — the [1100,1800] OCR-width optimization
  regressed nothing). frontend tsc + next build clean.
- **Perf fix:** `/predict` OCR was ~23s (EasyOCR at 1800px ×2 on CPU); bounded
  to [1100,1800] → ~2.4x faster on the low-res path, same accuracy. Needs a
  uvicorn restart to take effect on a running backend.
- **Lesson recorded:** never run CPU-heavy model passes (eval/calibration/full
  suite) in the background while the user has the local app open — it starves
  the live backend (made /predict, /chat, /security-pattern appear "stuck").
- **State:** all committed locally; ready to push to origin/main on request.



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
