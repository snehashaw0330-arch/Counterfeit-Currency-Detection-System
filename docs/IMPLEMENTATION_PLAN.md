# Counterfeit Currency Detection — Implementation Plan

Status as of 2026-05-16. This plan is the contract for the next
working session. Execute phases in order. Do not skip ahead.

---

## 1. Context and current state

### What already works

- FastAPI backend at `backend/main.py` with `/predict` endpoint
- Next.js frontend at `frontend/app/page.tsx`
- MobileNetV2 classifier loaded from `models/mobilenet_counterfeit_detector.keras`
- Forensic pipeline at `backend/forensic.py` with 8 checks:
  - structural_sanity ✓ (catches blank/noise/half-cropped)
  - uv_light_detection ✓
  - watermark_detection ✓
  - gandhi_face_analysis ✓
  - security_thread_detection ✓
  - colour_palette_integrity ✓ (separator score 15/20 on benchmark)
  - 19 unit tests passing
- Diagnostic harness at `tests/diagnostic_harness.py` against
  10 verified-real + 10 negative-control samples
- Combined verdict logic (REAL / SUSPICIOUS / FAKE) with structural
  sanity as hard gate

### What is broken / weak

- `extract_serial_number` — Tesseract misreads stylised banknote
  fonts. On the user's actual `images.note 500.jpg.jpeg` it
  returns `5BM 000793` (the digit `9` is read as `S` then
  normalised to `5`). EasyOCR returns the correct `9BM 000793`
  at 82% confidence.
- `classify_denomination` — Tesseract reads the `₹` symbol as
  `9`, so `₹500` becomes `900`. EasyOCR reads `8500` from which
  we strip the leading `8` to get `500`.
- User-reported failures on 10 hand-tested phone photos:
  both OCR and denomination return `FAIL` on the majority.

### Why we are switching the OCR engine

- Tesseract is designed for clean printed documents in standard
  fonts. Indian banknote serials and the `₹` symbol are
  outside that distribution.
- EasyOCR uses a deep-learning detector + recogniser and
  handles stylised fonts and the `₹` glyph correctly. Verified
  on the user's actual failing input.
- The cost is one-time: roughly 500 MB of PyTorch wheels into
  the venv plus ~64 MB of model weights downloaded on first
  use. EasyOCR is already installed in the venv at the time
  this plan was written.

---

## 2. Phase A — EasyOCR-based serial + denomination

### Goal

The two features the user is most upset about — OCR Serial
Number and Denomination Classification — must work on phone-
camera photos of real Indian banknotes.

### Reality check on the test fixtures

Two distinct quality grades of input matter for a production
system, and they should be scored separately:

- **Clean scans** (Wikipedia-grade): `tests/sample_notes/real/`
  already has 10 such samples — obverse + reverse for
  10, 20, 50, 200, 2000.
- **Phone photos** (handheld, tilted, background, varied
  resolution): these are the inputs the user actually cares
  about and previously failed on. We have 5 such photos in
  `C:\Users\swadh\Downloads\` from today that we lift into the
  repo under `tests/sample_notes/real_phone/`.

### Coverage matrix (after A.0a)

| Denom | Clean obv | Clean rev | Phone obv | Notes |
|-------|-----------|-----------|-----------|-------|
| 10    | ✓         | ✓         | —         | phone gap |
| 20    | ✓         | ✓         | specimen  | specimen has zero-serial |
| 50    | ✓         | ✓         | ✓ `1DV 708101` | |
| 100   | —         | —         | —         | **gap — see A.0c** |
| 200   | ✓         | ✓         | —         | phone gap |
| 500   | —         | —         | ✓✓ `9BM 000793` + `4AA 737195` | the `4AA` handheld is a Phase B goldmine (sky background, tilt) |
| 2000  | ✓         | ✓         | ✓ `3DM 401764` | |

Known-serial ground truth on the phone set: 4 samples
(50, 500, 500, 2000) — strong enough to make exact-string
serial assertions meaningful.

### Tasks

**A.0a — Lock today's Downloads photos into the repo.**

Create `tests/sample_notes/real_phone/` and copy in:

| Source (Downloads) | Destination | Ground truth |
|--------------------|-------------|--------------|
| `images.note 500.jpg.jpeg` | `real_500_phone_obv.jpg`           | denom 500, serial `9BM 000793` |
| `1 (3).jpg.jpeg`           | `real_500_phone_handheld_obv.jpg`  | denom 500, serial `4AA 737195` |
| `1 (42).jpg.jpeg`          | `real_2000_phone_obv.jpg`          | denom 2000, serial `3DM 401764` |
| `6.jpg.jpeg`               | `real_50_phone_obv.jpg`            | denom 50, serial `1DV 708101` |
| `1.jpg.jpeg`               | `real_20_phone_specimen_obv.jpg`   | denom 20, serial `null` (specimen) |

**A.0b — Write `tests/sample_notes/manifest.json`.**

Single source of truth for ground truth. Schema:

```json
{
  "<filename>": {
    "denom":  "10|20|50|100|200|500|2000",
    "side":   "obverse|reverse|specimen",
    "serial": "AAA 123456" | null,
    "source": "wikipedia|user_phone|synthetic_fake",
    "grade":  "clean|phone|fake"
  }
}
```

The Phase A pytest reads this file rather than hardcoded
paths, so adding a new sample just means dropping the file
in and appending one manifest line — no test code changes.

**A.0c — Resolve the Rs 100 gap.**

We have zero Rs 100 samples in the repo and no Rs 100 photo
in Downloads. Either:
- The user supplies a Rs 100 photo (preferred), or
- We pull obverse + reverse from Wikipedia following the
  same convention as the other clean samples in
  `tests/sample_notes/real/`.

**Open question for the user — do not begin A.1 until
resolved.**

**A.1 — EasyOCR singleton (extend what is already in
`backend/forensic.py`).**

The singleton exists at lines 26-50. Add:
- A public `warmup_ocr()` function that calls
  `_get_easyocr_reader()` and discards the result, so the
  FastAPI startup hook can pre-pay the ~3 s init cost.
- A constant `_ALNUM_SPACE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ
  0123456789 "` reused by both call sites.

**A.2 — Replace `extract_serial_number` with the EasyOCR
path.**

- Call `reader.readtext(image, allowlist=_ALNUM_SPACE,
  detail=1)` once per image. EasyOCR handles its own
  preprocessing — no CLAHE, no Otsu, no PSM permutations.
- For each `(bbox, text, conf)` tuple, try to match
  `^([A-Z0-9]{3})\s?([0-9OISB]{6,7})$` (single-token serial)
  AND pair adjacent tokens on the same horizontal band for the
  two-token `<prefix> <digits>` form.
- Apply the existing `_normalize_prefix` and
  `_normalize_digits` so OCR confusion fixes (`O→0`, `S→5`,
  `B→8`) still apply.
- Real banknotes carry the serial twice. Vote: count
  occurrences first, summed confidence second. A serial that
  appears twice wins over a higher-confidence solo read.
- Return shape unchanged: `{status, details, value}`.
- If `_get_easyocr_reader()` returns `None`, fall through to
  the existing Tesseract implementation (rename it
  `_extract_serial_number_tesseract` and keep it intact).

**A.3 — Replace `classify_denomination` with the EasyOCR
path.**

- Reuse the same cached reader, digit-only allowlist
  (`"0123456789"`).
- Run `_denom_candidates()` over each token, accumulate
  summed confidence per candidate denomination.
- Keep the existing `_palette_match` tiebreak so a `200`
  misread on a Rs 500 stone-grey note resolves to `500`.
- Return shape unchanged.
- Same Tesseract-fallback discipline as A.2: rename the
  current body to `_classify_denomination_tesseract`.

**A.4 — Pre-warm the OCR engine at FastAPI startup.**

In `backend/main.py`, add a startup event that calls
`forensic.warmup_ocr()` in a background thread so the first
real `/predict` request doesn't pay the cold-start latency. If
warmup fails (no internet on first ever run, model download
blocked), log and continue — the lazy path still works.

**A.5 — Manifest-driven pytest at `tests/test_phase_a_ocr.py`.**

Test cases (all read `manifest.json`, no hardcoded paths):

- `test_phone_serial_exact_match` — for every phone sample
  whose `serial` ground truth is non-null (currently 4:
  50, 500x2, 2000), assert `extract_serial_number` returns
  exactly that string. Allow ≤ 1 failure out of 4.
- `test_phone_denom_exact_match` — for every phone sample
  (5), assert `classify_denomination` returns the manifest
  denom. Allow ≤ 1 failure out of 5.
- `test_clean_obverse_denom_match` — for the 5 clean
  obverse samples, assert denom match. Allow ≤ 1 failure.
- `test_clean_obverse_serial_format` — for the 5 clean
  obverse samples, assert the serial value matches
  `^[A-Z0-9?]{3} [0-9]{6,7}$` (we don't know the
  Wikipedia-image ground truth, only the format). Allow
  ≤ 1 format failure.
- `test_fakes_no_spurious_serials` — for the 10 negative-
  control samples, assert ≤ 2 produce a non-null serial
  value.

Move the old `tests/test_easyocr_500.py` to
`scripts/probe_easyocr.py` — it's still useful as a manual
debugging script.

**A.6 — `requirements.txt` and setup docs.**

- Add `easyocr>=1.7,<2` to `requirements.txt` (currently
  installed but unpinned in the requirements file).
- Add a `docs/SETUP_COMMANDS.txt` note that the first launch
  downloads ~64 MB of EasyOCR detector + recogniser weights
  to `~/.EasyOCR/model/`. This is one-time per machine.

### Acceptance criteria for Phase A

All five must hold to commit and proceed to Phase B:

- `tests/test_phase_a_ocr.py` passes in full (5 tests).
- All existing 19 unit tests in `tests/test_forensic.py`
  still pass.
- Phone-photo grade: ≥ 3 of 4 known-serial phone samples
  produce the exact serial string; ≥ 4 of 5 phone samples
  produce the correct denomination.
- `tests/diagnostic_harness.py` on the expanded sample set
  reports no regression vs the pre-Phase-A baseline (clean
  reals → REAL ≥ 8/10; fakes → FAKE+SUSPICIOUS ≥ 8/10).
- One manual smoke run by the user end-to-end through the
  Next.js frontend on at least 2 phone photos of their own
  notes, both producing the correct serial AND denomination.

If any criterion fails, do not move to Phase B.

### Out of scope for Phase A

- Frontend changes — the response shape is unchanged
- Retraining the MobileNetV2 model
- New forensic checks (auto-crop is Phase B)
- UI tweaks

---

## 3. Phase B — Auto-crop the note from background

### Goal

Phone photos often include desk, hand, table edges, etc. The
note itself can occupy as little as 30% of the frame. Cropping
to the note before feature extraction improves every downstream
check.

### Tasks

1. New helper in `backend/forensic.py`:
   `_locate_note(bgr_image)` that returns a tighter BGR crop
   containing just the banknote, or the original image if no
   plausible note region is found.
   - Convert to grayscale, run Canny edge detection.
   - Find external contours, filter for quadrilateral shapes
     with aspect ratio between 1.8 and 3.0 (Indian notes are
     roughly 2.2:1).
   - Pick the largest qualifying contour.
   - Apply a perspective transform so the result is rectified
     to a canonical horizontal orientation.
   - If nothing plausible is found, return the original image
     unchanged. Never raise.

2. Wire the helper into `run_forensic_pipeline` so every
   forensic check sees the cropped note rather than the raw
   frame.

3. Add unit tests for `_locate_note`:
   - Returns a smaller image when fed a synthetic note on a
     larger background.
   - Returns the input unchanged when fed a pure-note image
     (no background to crop).
   - Returns the input unchanged on garbage input (blank,
     noise).

### Acceptance criteria for Phase B

- Two new pytest tests pass.
- Re-running the harness on the existing 20-sample set
  produces a verdict count no worse than Phase A baseline
  (real → REAL ≥ 8, fake → FAKE + SUSPICIOUS ≥ 8).
- Manual test on at least three phone-camera photos with
  visible background: cropped output is visibly the note
  region.

### Out of scope for Phase B

- Multi-note images (more than one banknote in frame)
- Heavily folded or torn notes
- Tilt correction beyond perspective-rectify of a flat note

---

## 4. Phase C — Reactive polish

Phase C work is opened in response to specific user-reported
failures. Each sub-phase ships with its own commit, fixtures,
and tests.

### Phase C-1 — Banknote proportion check (new forensic feature)

**Why:** A counterfeit can be a real-note image that has been
digitally stretched/squashed to disguise origin, or a printed
fake on incorrect-size paper. RBI banknotes have specific
official dimensions per denomination — comparing the detected
note quad's aspect against the canonical aspect for the OCR'd
denomination gives a quantitative signal.

**Canonical RBI dimensions (mm):**

| Denom | mm        | Aspect |
|-------|-----------|--------|
| ₹10   | 123 × 63  | 1.952  |
| ₹20   | 129 × 63  | 2.048  |
| ₹50   | 135 × 66  | 2.045  |
| ₹100  | 142 × 66  | 2.152  |
| ₹200  | 146 × 66  | 2.212  |
| ₹500  | 150 × 66  | 2.273  |
| ₹2000 | 166 × 66  | 2.515  |

**Tasks:**

1. Refactor `_locate_note` into two pieces:
   - `_detect_note_quad(image)` — returns `{found, quad, aspect}`
     or `None`. Pure detection, no warp.
   - `_locate_note(image)` — calls the detector then perspective-
     rectifies the quad. Behaviour unchanged.
2. Add `_RBI_DIMENSIONS` constants and a derived expected-aspect
   table.
3. Implement `analyze_proportions(image, denomination)`:
   - PASS: detected aspect within 8% of canonical
   - FAIL: deviation ≥ 8% (likely stretching or wrong-size paper)
   - INFO: quad not detectable OR denomination unknown
   - `value` field carries `actual_aspect`, `expected_aspect`,
     and `deviation_pct` so the frontend can render the number
4. Wire into `run_forensic_pipeline` after
   `classify_denomination` (proportion check consumes the denom
   result). Pass the *original* image (not the auto-cropped
   one) so the quad-detector sees background to measure against.
5. Add `proportion_analysis` to `EXPECTED_KEYS` in
   `tests/test_forensic.py` so the shape contract stays honest.
6. New pytest module `tests/test_phase_c_proportions.py`:
   - PASS on a synthetic correctly-proportioned note
   - FAIL on a 30% horizontally-stretched note
   - INFO when no quad is detectable (blank input)
   - INFO when denomination is None
7. Add a synthetic stretched fixture
   `real_50_phone_stretched_obv.jpg` (programmatic 30% horizontal
   stretch of `real_50_phone_obv.jpg`) and a manifest entry with
   `grade: fake`, `fake_type: digital_stretch`.

**Acceptance criteria:**

- All new Phase C-1 tests pass
- Existing 25/25 still pass
- Diagnostic harness: no regression on real bucket; new stretched
  fixture detected as FAIL by `proportion_analysis`
- Frontend response gains a `proportion_analysis` key — existing
  keys unchanged

### Phase C-2 — TBD reactive items

Possible follow-ups (pick based on what's still failing):
1. Replace EasyOCR with PaddleOCR if Phase A still misses
   readable serials. PaddleOCR has the same `(bbox, text,
   conf)` shape so the call sites only swap the engine.
2. Add a serial cross-check: for any reading we surface, look
   up whether the encoded RBI inspection date (embedded in
   the serial prefix) is plausible. Wrong-format serials get
   flagged.
3. Frontend: highlight detected suspicious regions on the
   uploaded image (bounding boxes over failed forensic
   checks).
4. Add a `/diagnose` endpoint that returns raw OCR text and
   per-check intermediate values, useful for the user when
   the system gets a wrong answer.

### Acceptance criteria for Phase C-2+

Decided per task. The Phase C work order is reactive — the user
brings a failing real note, we fix that specific case.

---

## 5. What is explicitly NOT in this plan

- UV / infrared imaging support — requires hardware
- Retraining MobileNetV2 — needs a labelled banknote dataset
  we don't have
- Multi-currency support — Indian notes only
- 100% accuracy guarantee — physically impossible without
  hardware sensors. The honest target is ≥ 85% on phone
  photos of common denominations.

---

## 6. Working agreement

- Each phase ends with a commit and a manual test by the user.
- If Phase A acceptance criteria fail, we do not implement
  Phase B until A is fixed.
- The diagnostic harness (`tests/diagnostic_harness.py`)
  is the source of truth for objective numbers. Anything the
  user reports as "broken on this photo" gets added as a
  named sample under `tests/sample_notes/` so we never
  regress on it.
- Do not change response shapes, frontend types, or public
  forensic-pipeline keys without updating the frontend in
  the same commit.

---

## 7. Files this plan will touch

Phase A:
- `backend/forensic.py` — EasyOCR rewrites of
  `extract_serial_number` and `classify_denomination`,
  Tesseract bodies kept as `_*_tesseract` fallbacks,
  `warmup_ocr()` helper, shared `_ALNUM_SPACE` constant
- `backend/main.py` — FastAPI startup hook calling
  `warmup_ocr()` in a background thread
- `requirements.txt` — pin `easyocr>=1.7,<2`
- `tests/sample_notes/real_phone/` — 5 new phone-photo
  fixtures (copied from user Downloads, renamed)
- `tests/sample_notes/real/real_100_obv.jpg` +
  `real_100_rev.jpg` — fills the Rs 100 gap (subject to
  A.0c resolution)
- `tests/sample_notes/manifest.json` — ground-truth source
  of truth for the whole fixture tree
- `tests/test_phase_a_ocr.py` — new manifest-driven pytest
  module (5 tests)
- `scripts/probe_easyocr.py` — old `test_easyocr_500.py`
  relocated as a debug script
- `docs/SETUP_COMMANDS.txt` — note that first run downloads
  the EasyOCR detector + recogniser models (~64 MB)

Phase B:
- `backend/forensic.py` — `_locate_note` helper,
  `run_forensic_pipeline` wiring
- `tests/test_forensic.py` — new tests

Phase C:
- TBD per task
