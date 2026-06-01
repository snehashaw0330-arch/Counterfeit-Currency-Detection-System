"""
End-to-end system evaluation.

Runs every image in dataset/index.json through the REAL FastAPI
/predict endpoint (via TestClient, so the exact production verdict
logic — CNN + forensic + classical second opinion + combined
REAL/SUSPICIOUS/FAKE — is exercised, no duplication) and reports a
confusion matrix of the final verdict against ground truth.

This is the honest "how good is the whole product?" number, as
opposed to docs/BENCHMARK.md which scores the classical models on
features in isolation.

Run:
    venv\\Scripts\\python.exe scripts\\evaluate_system.py [--split test|train|all]

Interpretation (counterfeit-detector framing):
  genuine note -> REAL        = correct (cleared)
  genuine note -> SUSPICIOUS  = over-cautious (soft miss)
  genuine note -> FAKE        = FALSE POSITIVE (bad: rejects real money)
  fake note    -> FAKE        = correct (caught)
  fake note    -> SUSPICIOUS  = flagged for review (partial catch)
  fake note    -> REAL        = FALSE NEGATIVE (dangerous: passes a fake)
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INDEX = os.path.join(ROOT, "dataset", "index.json")
_VERDICTS = ("REAL", "SUSPICIOUS", "FAKE")


def main(split):
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    with open(INDEX, "r", encoding="utf-8") as fh:
        samples = json.load(fh)["samples"]
    if split != "all":
        samples = [s for s in samples if s["split"] == split]

    # confusion[ground_truth][verdict]
    confusion = {
        "genuine": {v: 0 for v in _VERDICTS},
        "fake": {v: 0 for v in _VERDICTS},
    }
    errors = []

    print(f"Evaluating {len(samples)} images (split={split}) through /predict...")

    for i, s in enumerate(samples, 1):
        path = os.path.join(ROOT, s["path"].replace("/", os.sep))
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            r = client.post(
                "/predict",
                files={"file": (os.path.basename(path), data, "image/jpeg")},
            )
            body = r.json()
            verdict = body.get("prediction")
        except Exception as exc:
            verdict = None
            errors.append((s["path"], str(exc)))

        if verdict in _VERDICTS:
            confusion[s["label"]][verdict] += 1
        else:
            errors.append((s["path"], f"no verdict: {body.get('message') if 'body' in dir() else verdict}"))
        if i % 10 == 0:
            print(f"  ...{i}/{len(samples)}")

    # ---- report ----
    print("\n=== End-to-end verdict confusion (rows = ground truth) ===")
    print(f"{'':10} | {'REAL':>10} {'SUSPICIOUS':>12} {'FAKE':>8}")
    print("-" * 46)
    for gt in ("genuine", "fake"):
        c = confusion[gt]
        print(f"{gt:10} | {c['REAL']:>10} {c['SUSPICIOUS']:>12} {c['FAKE']:>8}")

    n_gen = sum(confusion["genuine"].values())
    n_fake = sum(confusion["fake"].values())

    # Headline metrics (guard divide-by-zero)
    def pct(a, b):
        return f"{(100.0 * a / b):.1f}%" if b else "n/a"

    gen_cleared = confusion["genuine"]["REAL"]
    gen_false_pos = confusion["genuine"]["FAKE"]
    fake_caught = confusion["fake"]["FAKE"] + confusion["fake"]["SUSPICIOUS"]
    fake_passed = confusion["fake"]["REAL"]

    print("\n=== headline (counterfeit-detector framing) ===")
    print(f"  genuine cleared as REAL      : {gen_cleared}/{n_gen}  ({pct(gen_cleared, n_gen)})")
    print(f"  genuine wrongly FAKE (FP)    : {gen_false_pos}/{n_gen}  ({pct(gen_false_pos, n_gen)})")
    print(f"  fake flagged (FAKE+SUSP)     : {fake_caught}/{n_fake}  ({pct(fake_caught, n_fake)})")
    print(f"  fake passed as REAL (FN)     : {fake_passed}/{n_fake}  ({pct(fake_passed, n_fake)})  <- most dangerous")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for p, e in errors[:8]:
            print(f"  {p}: {e}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train", "all"], default="all")
    args = ap.parse_args()
    raise SystemExit(main(args.split))
