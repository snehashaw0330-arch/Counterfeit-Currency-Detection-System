# Phase R + S — Progress & Remaining Work

Tracking the 11-item requirement (serial-guilloché generation + verification,
digital PUF, foreign/polymer multi-currency detection). Contract lives in
[PROJECT_SCOPE.md](PROJECT_SCOPE.md) (Phase R, Phase S, two-agent ownership);
this file is the living checklist we work through **one item at a time**.

**Last updated:** 2026-06-28

---

## Requirement scorecard

| # | Requirement | Status |
|---|---|---|
| 1 | Guilloché generator from serial number | ✅ Done |
| 2 | Same serial → same unique pattern | ✅ Done |
| 3 | Guilloché verification (generated vs extracted) | ✅ Done |
| 4 | Digital PUF for unique note authentication | ✅ Done |
| 5 | Support for foreign polymer banknotes | 🟡 Partial (detection + polymer cues; no foreign counterfeit verdict) |
| 6 | Expand dataset with multi-country polymer notes | ✅ Done (BDT 672-img slice normalized + validated; BankNote-Net + genuine polymer refs; polymer-counterfeit data n/a) |
| 7 | Automatic country & currency detection | ✅ Done (GBP still weak/UNKNOWN) |
| 8 | Multi-currency **counterfeit** detection | 🟢 Done for INR + BDT (BDT model: 0.909 image / 0.955 note acc); AUD/CAD/GBP/PHP impossible (no public fake data) |
| 9 | Polymer-specific forensic/security checks | ✅ Done (window + sheen; tested on genuine refs) |
| 10 | Update report (guilloché, PUF, polymer) | 🔴 Not done |
| 11 | Evaluate on Indian + foreign datasets | 🔴 Not done |

---

## ✅ Shipped (done + tested, INR `/predict` path contract-stable)

**Track 1 — self-authenticating note scheme**
- **R.1** `backend/security_pattern.py` `secure_note_token/png` (serial-seeded
  guilloché + canonical serial + verification CODE, crisp OCR-legible header) ·
  `GET /secure-note/generate` · `tests/test_phase_r_secure_note.py` (9).
- **R.2** `backend/secure_note.py` `verify_secure_note` (regenerate-and-compare
  at native scale, calibrated 3-band AUTHENTIC/TAMPERED/UNVERIFIED, OCR-trust
  rule) · `POST /secure-note/verify` · `tests/test_phase_r_verify.py` (13) +
  `_verify_ocr.py` (3, slow/skippable).
- **R.3** `backend/puf.py` digital PUF (texture-fingerprint Hamming hash,
  calibrated 0.25; same-capture ≤0.04, different ≥0.36; git-ignored JSON
  registry) · `POST /puf/enroll` + `POST /puf/verify` ·
  `tests/test_phase_r_puf.py` (12).

**Track 2 — foreign/currency integration**
- `backend/country.py` (Codex) `detect_country` + `backend/polymer.py` (Codex)
  polymer cue checks · `tests/test_country.py` + `tests/test_polymer.py` (13).
- **S.2/S.3** `backend/foreign_routing.py` wires both into `/predict` behind a
  gate (detector runs ONLY when the note isn't identified as INR → zero cost on
  the common path; confident foreign → country/polymer fields + verdict forced
  UNVERIFIED) · `POST /detect-country` · `tests/test_phase_s_integration.py` (6).

**Frontend (Next 16 / React 19, tsc clean)**
- `SecureNoteStudio` — generate + verify + PUF enroll/verify panels.
- `CountryDetectionPanel` — surfaces detected foreign currency + polymer cue in
  `/predict` results. `PredictResponse` type extended.

---

## ⏭️ Remaining work — address one at a time (suggested order)

### T1 — Finalize the foreign dataset (item 6) — ✅ DONE
- `scripts/fetch_jaaltaka.py` rewritten to the locked layout
  `dataset/foreign/bdt/security_crops/{real,fake}/note_XXX_k.jpg` + JSON sidecars
  (country/currency/substrate/denomination/side/label/source + note_id/segment).
- `scripts/build_dataset.py` now **skips `dataset/foreign/`**, so BDT crops never
  pool into the INR whole-note classifier.
- Pulled the full slice: **672 BDT images (336 real / 336 fake), 56 notes/class**,
  every image with a sidecar.
- **Acceptance met:** `validate_foreign_dataset.py` → "looks good"; INR index
  unchanged at 65 (42/23), zero foreign leakage.

### T2 — BDT counterfeit model (the real item 8) — ✅ DONE
- `scripts/train_bdt_counterfeit.py` — mirrors the INR classical pipeline (same
  `backend.features` extractor + 4 techniques), **group-aware split by note**
  (`<label>_<note_id>`, no crop leakage), no augmentation (672 real images).
- Held-out test (132 images / 22 unseen notes): **Random Forest** (best by CV
  macro-F1 0.888) → **0.909 image acc / 0.955 note acc**; SVM 0.924 image.
  `models/bdt/metrics.json` (tracked; binaries git-ignored).
- `backend/bdt_classifier.py` — loads best model, 3-band REAL/SUSPICIOUS/FAKE.
  `backend/foreign_routing.py` runs it for a confident BDT note → real
  counterfeit verdict in `/predict`; frontend shows it.
- **Acceptance met:** honest group-aware metrics recorded; live path verified
  (real→REAL, fake→FAKE); 60 fast tests + 6 API tests green; INR path unchanged.

### T3 — Evaluation: INR vs foreign (item 11)
- Per-currency confusion matrices + comparison table (INR counterfeit, BDT
  counterfeit, foreign currency-ID accuracy). Script-driven + reproducible.
- **Acceptance:** numbers committed to `docs/BENCHMARK.md` / report; method
  honest (no test peeking, group-aware).

### T4 — Update the report (item 10)
- `docs/REPORT.md` + `docs/GENAI_EXPLAINED.md`: add the secure-note scheme
  (guilloché generation + verification), the digital PUF, country detection, and
  polymer checks — with the honest limits below. Tick Phase R/S acceptance in
  `PROJECT_SCOPE.md`.

### T5 — Polish (optional, as time allows)
- `country.py` OCR: reuse forensic's EasyOCR singleton instead of a new Reader
  per call + drop Tesseract-first (aligns with the project's EasyOCR migration;
  removes per-call latency) — coordinate with Codex (his lane).
- Improve GBP detection (currently stays UNKNOWN — honest but a miss).
- Run a full `next build` before the demo.

---

## Honest limits (state these in the report — do not paper over)

- **No foreign counterfeit detection beyond BDT.** Polymer-counterfeit image
  datasets are essentially non-public, so AUD/CAD/GBP/PHP get currency
  identification + polymer cues only — never a counterfeit verdict. BDT is the
  one foreign currency we can do counterfeit detection for (JaalTaka has real
  fakes).
- **The secure-note scheme is a proof-of-concept**, not verification of real
  banknotes (real notes carry no serial-derived guilloché or enrolled PUF).
- **PUF is a software proxy** — robust to re-encoding and discriminates
  different captures, but matching across genuinely different phone photos of the
  same physical note (lighting/angle) is the hard real-world part; needs
  enrollment to verify against.
- **BankNote-Net** is embeddings-only + genuine-only + has no encoder shipped →
  kept as an offline benchmark, not the live country-detection path.

---

## Operational notes
- **Restart `uvicorn`** to load the new endpoints.
- `tsc --noEmit` passes; run a full `next build` before demoing.
- `main.py` carries a concurrent edit (`run_in_threadpool`, `classical_status`)
  from a parallel session — harmless, coexists, tests green.
