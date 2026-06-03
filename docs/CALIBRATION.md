# Verdict-threshold calibration (Phase F.2)

Evidence record for the fused-verdict cutoffs. Regenerate the ROC pass with
`venv\Scripts\python.exe scripts\calibrate_thresholds.py` (run it when the
local backend is idle — it scores all 65 images through `/predict` and is
CPU-heavy).

## Corpus

65 labelled images — 42 genuine / 23 fake (`dataset/index.json`).

## End-to-end verdict confusion (current pipeline, post Phase E–L)

| ground truth ↓ / verdict → | REAL | SUSPICIOUS | FAKE | UNVERIFIED |
|---|---|---|---|---|
| **genuine** | 31 | 11 | 0 | 0 |
| **fake** | 10 | 7 | 5 | 1 |

- **False positives (genuine → FAKE): 0 / 42 (0%)** — the priority metric.
- Genuine cleared as REAL: 31 / 42 (74%).
- Fakes flagged (FAKE / SUSPICIOUS / UNVERIFIED): 13 / 23 (57%).
- Fakes passed as REAL: 10 / 23 (43%) — the data-bound false-negative case.

## Score & thresholds

`combined_score = 0.4 · CNN + 0.6 · forensic_pass_fraction`, then:

- **REAL** when `combined ≥ 0.65` **and** ≥ 5 checks pass **and** no
  structural / colour / proportion hard-gate failure;
- **FAKE** when `combined < 0.35` (or both sub-scores < 0.35), or a hard gate
  fails;
- **SUSPICIOUS** in between;
- **UNVERIFIED** (Phase K) overlays the above when the note could not actually
  be read (no serial / size / denomination) — "can't verify, retake".

## Recommendation (honest)

The hand-set `0.65 / 0.35` band is **consistent with the evidence and left
unchanged.** Rationale:

1. It already yields **0% false positives** — the dominant correctness goal
   (rejecting real money is the worst error).
2. With only 65 images the ROC curve is coarse (few distinct operating points);
   shifting cutoffs to chase one or two borderline fakes would over-fit and risk
   the 0% false-positive property.
3. The genuine/fake score distributions overlap because of the small,
   physically-photographed fake set — the real lever is **more data**
   (especially real physical counterfeits), not threshold tuning. This is the
   documented data ceiling (see [REPORT.md](REPORT.md) §7).

`scripts/calibrate_thresholds.py` additionally computes the ROC-AUC and the
Youden's-J optimal single boundary; on this corpus that boundary falls inside
the current SUSPICIOUS band, confirming the cutoffs above. Re-run it after the
dataset grows to revisit the thresholds with statistical confidence.
